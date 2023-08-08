from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g")

sequences_list = []

nts = ['A', 'C', 'G', 'T', 'N']

# make all possible 6-mer sequences
for nt1 in nts:
    for nt2 in nts:
        for nt3 in nts:
            for nt4 in nts:
                for nt5 in nts:
                    for nt6 in nts:
                        sequences_list.append(nt1 + nt2 + nt3 + nt4 + nt5 + nt6)

sequences_list = list(set(sequences_list))

sequences_list = sequences_list[:500]

all_embeddings = []

# loop through 150 sequences at a time
i = 0
while i < len(sequences_list):
    sequences = [''.join(sequences_list[i:i+150])]
    print(len(sequences))
    # Create a dummy dna sequence and tokenize it
    tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt")["input_ids"]
    print(f"Tokens ids: {tokens_ids}")

    # Compute the embeddings
    attention_mask = tokens_ids != tokenizer.pad_token_id
    torch_outs = model(
        tokens_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True
    )

    # Compute sequences embeddings
    embeddings = torch_outs['hidden_states'][-1].detach().numpy()
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings per token: {embeddings}")
    # remove the first codon
    embeddings = embeddings[:, 1:, :]
    # remove first dim
    # embeddings = torch.squeeze(embeddings, axis=0)
    embeddings = embeddings[0]
    print(f"Embeddings shape: {embeddings.shape}")

    all_embeddings.append(embeddings)

    i += 150

# concatenate all embeddings
all_embeddings = np.concatenate(all_embeddings, axis=0)

# make dict from sequences_list and embeddings
embeddings_dict = dict(zip(sequences_list, all_embeddings))

# save dict
import pickle
with open('idai_epa_embeds.pkl', 'wb') as f:
    pickle.dump(embeddings_dict, f)

