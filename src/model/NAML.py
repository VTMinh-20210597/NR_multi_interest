import torch
from torch import nn
import torch.nn.functional as F

from .model_utils import AttentionPooling

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Learnable Positional Encoding
        
        Args:
            d_model (int): Dimension of the model's embeddings
            max_len (int): Maximum sequence length to support
        """
        super(PositionalEncoding, self).__init__()
        
        # Create learnable positional embeddings
        self.positional_embedding = nn.Parameter(torch.zeros(max_len, d_model))
        
        # Initialize the embedding using xavier uniform initialization
        nn.init.xavier_uniform_(self.positional_embedding)

    def forward(self, x):
        """
        Add positional embeddings to input tensor
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
        
        Returns:
            Tensor: Input tensor with positional embeddings added
        """
        # Slice positional embeddings to match input sequence length
        seq_len = x.size(1)
        
        # Add positional embeddings to input
        return x + self.positional_embedding[:seq_len, :].unsqueeze(0)

class NewsEncoder(nn.Module):
    def __init__(self, args, embedding_matrix, num_category, num_subcategory):
        super(NewsEncoder, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.drop_rate = args.drop_rate
        self.num_words_title = args.num_words_title
        self.use_category = args.use_category
        self.use_subcategory = args.use_subcategory
        if args.use_category:
            self.category_emb = nn.Embedding(num_category + 1, args.category_emb_dim, padding_idx=0)
            self.category_dense = nn.Linear(args.category_emb_dim, args.news_dim)
        if args.use_subcategory:
            self.subcategory_emb = nn.Embedding(num_subcategory + 1, args.category_emb_dim, padding_idx=0)
            self.subcategory_dense = nn.Linear(args.category_emb_dim, args.news_dim)
        if args.use_category or args.use_subcategory:
            self.final_attn = AttentionPooling(args.news_dim, args.news_query_vector_dim)
        self.cnn = nn.Conv1d(
            in_channels=args.word_embedding_dim,
            out_channels=args.news_dim,
            kernel_size=3,
            padding=1
        )
        self.attn = AttentionPooling(args.news_dim, args.news_query_vector_dim)

    def forward(self, x, mask=None):
        '''
            x: batch_size, word_num
            mask: batch_size, word_num
        '''
        title = torch.narrow(x, -1, 0, self.num_words_title).long()
        word_vecs = F.dropout(self.embedding_matrix(title),
                              p=self.drop_rate,
                              training=self.training)
        context_word_vecs = self.cnn(word_vecs.transpose(1, 2)).transpose(1, 2)
        title_vecs = self.attn(context_word_vecs, mask)
        all_vecs = [title_vecs]

        start = self.num_words_title
        if self.use_category:
            category = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            category_vecs = self.category_dense(self.category_emb(category))
            all_vecs.append(category_vecs)
            start += 1
        if self.use_subcategory:
            subcategory = torch.narrow(x, -1, start, 1).squeeze(dim=-1).long()
            subcategory_vecs = self.subcategory_dense(self.subcategory_emb(subcategory))
            all_vecs.append(subcategory_vecs)

        if len(all_vecs) == 1:
            news_vecs = all_vecs[0]
        else:
            all_vecs = torch.stack(all_vecs, dim=1)
            news_vecs = self.final_attn(all_vecs)
        return news_vecs

class UserEncoder(nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.num_interests = 3  # New parameter to specify number of interests
        
        # Use multiple attention pooling layers for different interests
        self.interest_attns = nn.ModuleList([
            AttentionPooling(args.news_dim, args.user_query_vector_dim) 
            for _ in range(self.num_interests)
        ])
        
        self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)
        
        # Add positional encoding
        self.positional_encoding = PositionalEncoding(args.news_dim, max_len=args.user_log_length)

    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
            
            Returns: batch_size, num_interests, news_dim
        '''
        bz = news_vecs.shape[0]
        
        # Apply positional encoding
        news_vecs = self.positional_encoding(news_vecs)
        
        # Compute multiple interest vectors
        user_interests = []
        if self.args.user_log_mask:
            # If using log mask, apply different attention for each interest
            for attn in self.interest_attns:
                user_interest = attn(news_vecs, log_mask)
                user_interests.append(user_interest)
        else:
            # If not using log mask, use padding for each interest
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.args.user_log_length, -1)
            for attn in self.interest_attns:
                news_vecs_padded = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
                user_interest = attn(news_vecs_padded)
                user_interests.append(user_interest)
        
        # Stack the multiple interest vectors
        user_interests = torch.stack(user_interests, dim=1)
        return user_interests

class Model(torch.nn.Module):
    def __init__(self, args, embedding_matrix, num_category, num_subcategory, **kwargs):
        super(Model, self).__init__()
        self.args = args
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=args.freeze_embedding,
                                                      padding_idx=0)

        self.news_encoder = NewsEncoder(args, word_embedding, num_category, num_subcategory)
        self.user_encoder = UserEncoder(args)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, history, history_mask, candidate, label):
        '''
            history: batch_size, history_length, num_word_title
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''
        batch_size, num_candidates, num_word_title = candidate.shape

        candidate_news = candidate.reshape(-1, num_word_title)
        candidate_news_vecs = self.news_encoder(candidate_news).reshape(-1, 1 + self.args.npratio, self.args.news_dim)

        history_news = history.reshape(-1, num_word_title)
        history_news_vecs = self.news_encoder(history_news).reshape(-1, self.args.user_log_length, self.args.news_dim)
        print(f"history_news_vecs: {history_news_vecs.shape}") #history_news_vecs: torch.Size([32, 50, 400])


        # User interests now will be batch_size, num_interests, news_dim
        user_interests = self.user_encoder(history_news_vecs, history_mask)
        print(f"user_interests.shape: {user_interests.shape}") #user_interests.shape: torch.Size([32, 3, 400])
        
        # Compute scores for each interest
        scores = torch.bmm(candidate_news_vecs, user_interests.transpose(1, 2))
        
        # You might want to aggregate scores (e.g., mean, max, etc.)
        score = scores.mean(dim=-1)
        
        loss = self.loss_fn(score, label)
        return loss, score