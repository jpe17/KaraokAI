import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperFeatureExtractor
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            # Truncate sequence to fit positional encoding
            x = x[:self.pe.size(0)]
            seq_len = self.pe.size(0)
        return x + self.pe[:seq_len, :]

class VoiceToNotesModel(nn.Module):
    def __init__(self, 
                 whisper_model_name="openai/whisper-base",
                 d_model=512,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_notes=500):
        super().__init__()
        
        # Whisper encoder (frozen)
        self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        
        # Freeze whisper parameters
        for param in self.whisper.parameters():
            param.requires_grad = False
        
        # IMPROVED: Better encoder projection with residual connection
        whisper_dim = self.whisper.config.d_model
        self.encoder_projection = nn.Sequential(
            nn.Linear(whisper_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Reduced dropout
            nn.Linear(d_model, d_model)
        )
        
        # Note vocabulary: 3 special tokens + 128 MIDI notes + time/duration buckets
        # [PAD], [START], [END], notes 0-127, time buckets 0-99, duration buckets 0-99
        self.vocab_size = 3 + 128 + 100 + 100  # 331 total
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        
        # IMPROVED: Better token embeddings with proper initialization
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_notes)
        
        # IMPROVED: Transformer decoder with balanced dropout
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,  # Use standard dropout, not doubled
            activation=F.gelu,  # Better activation than ReLU
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # IMPROVED: Controlled dropout layers
        self.embedding_dropout = nn.Dropout(dropout * 0.5)  # Reduced embedding dropout
        self.output_dropout = nn.Dropout(dropout * 0.5)     # Reduced output dropout
        
        # IMPROVED: Output heads with better architecture
        self.note_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 128)  # MIDI notes 0-127
        )
        self.time_head = nn.Linear(d_model, 100)  # Time buckets
        self.duration_head = nn.Linear(d_model, 100)  # Duration buckets
        
        self.d_model = d_model
        self.max_notes = max_notes
        
        # IMPROVED: Initialize weights for better convergence
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better convergence"""
        # Initialize token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def encode_audio(self, audio_features):
        """Encode audio using Whisper encoder with improved projection"""
        with torch.no_grad():
            encoder_outputs = self.whisper.encoder(audio_features)
        
        # IMPROVED: Better projection with residual connection
        encoder_hidden = self.encoder_projection(encoder_outputs.last_hidden_state)
        return encoder_hidden
    
    def create_target_mask(self, size):
        """Create causal mask for decoder"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 1
    
    def forward(self, audio_features, target_tokens=None, max_length=None):
        # Encode audio
        encoder_hidden = self.encode_audio(audio_features)
        
        if target_tokens is not None:
            # Training mode
            seq_len = target_tokens.size(1)
            target_mask = self.create_target_mask(seq_len).to(target_tokens.device)
            
            # IMPROVED: Better token embedding with proper scaling
            target_embedded = self.token_embedding(target_tokens) * math.sqrt(self.d_model)
            target_embedded = self.embedding_dropout(target_embedded)
            target_embedded = self.positional_encoding(target_embedded.transpose(0, 1)).transpose(0, 1)
            
            # Decoder forward pass
            decoder_output = self.transformer_decoder(
                target_embedded,
                encoder_hidden,
                tgt_mask=target_mask
            )
            
            # IMPROVED: Skip connection before output heads
            decoder_output = decoder_output + target_embedded  # Residual connection
            decoder_output = self.output_dropout(decoder_output)
            
            # Get predictions
            note_logits = self.note_head(decoder_output)
            time_logits = self.time_head(decoder_output)
            duration_logits = self.duration_head(decoder_output)
            
            return note_logits, time_logits, duration_logits
        else:
            # Inference mode
            return self.generate(encoder_hidden, max_length or self.max_notes)
    
    def generate(self, encoder_hidden, max_length):
        """Generate note sequence autoregressively"""
        batch_size = encoder_hidden.size(0)
        device = encoder_hidden.device
        
        # Start with start token
        generated = torch.full((batch_size, 1), self.start_token, device=device)
        
        for _ in range(max_length):
            # Get current length
            seq_len = generated.size(1)
            
            # Stop if we exceed positional encoding capacity
            if seq_len >= self.max_notes:
                break
                
            target_mask = self.create_target_mask(seq_len).to(device)
            
            # Embed current sequence
            target_embedded = self.token_embedding(generated) * math.sqrt(self.d_model)
            target_embedded = self.positional_encoding(target_embedded.transpose(0, 1)).transpose(0, 1)
            
            # Decoder forward pass
            decoder_output = self.transformer_decoder(
                target_embedded,
                encoder_hidden,
                tgt_mask=target_mask
            )
            
            # Get next token predictions
            next_note = self.note_head(decoder_output[:, -1:])
            next_time = self.time_head(decoder_output[:, -1:])
            next_duration = self.duration_head(decoder_output[:, -1:])
            
            # Sample next tokens (simplified - only use note prediction)
            next_note_token = torch.argmax(next_note, dim=-1)
            
            # Use note token directly
            next_token = next_note_token + 3  # Offset by special tokens
            
            # Check for end token
            if torch.any(next_token == self.end_token):
                break
                
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def tokenize_notes(self, notes_data):
        """Convert note data to tokens - SIMPLIFIED VERSION"""
        # notes_data: list of (start_time, duration, note) tuples
        tokens = [self.start_token]
        
        # SIMPLIFIED: Just use note tokens for now
        for start_time, duration, note in notes_data:
            tokens.append(int(note) + 3)  # Notes start at index 3
        
        tokens.append(self.end_token)
        return tokens
    
    def detokenize_notes(self, tokens):
        """Convert tokens back to note data - SIMPLIFIED VERSION"""
        notes = []
        tokens = tokens.cpu().numpy()
        
        print(f"Detokenizing tokens: {tokens[:20]}...")  # Debug
        
        # Skip start token
        i = 1
        while i < len(tokens):
            if tokens[i] == self.end_token:
                break
            
            # Check if we have a valid note token (3-130 range for notes)
            if tokens[i] >= 3 and tokens[i] < 131:
                note = tokens[i] - 3
                
                # Create notes with reasonable timing based on position
                start_time = (i - 1) * 0.25  # 0.25 second intervals for more realistic timing
                duration = 0.25  # Shorter duration
                
                if note >= 0 and note < 128:
                    notes.append((start_time, duration, note))
            
            i += 1
        
        print(f"Generated {len(notes)} notes")
        return notes 