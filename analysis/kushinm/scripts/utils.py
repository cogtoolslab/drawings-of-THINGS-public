import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import pearsonr

def SDI(data):
    """
    Calculate the Simpson's Diversity Index for a list of data.
    """
    N = np.sum(data)
    n = np.array(data)
    
    p = n / N
    return 1 - np.sum(p ** 2)


def mm_normalize(row, min_max_dict):
    min_recog = min_max_dict[row['uniqueID']]['min']
    max_recog = min_max_dict[row['uniqueID']]['max']
    if (max_recog - min_recog) == 0:
        return row['mean_accuracy']
    norm_acc = (row['mean_accuracy'] - min_recog) / (max_recog - min_recog)
    return norm_acc




def compute_similarities(df, spose_embeds, spose_cols, random=False):
    similarities = []
    for _, row in df.iterrows():
        if random:
            target_concept = np.random.choice(spose_embeds['concept'].unique())
        else:
            target_concept = row.uniqueID
        
        target_embed = spose_embeds[spose_embeds['concept'] == target_concept][spose_cols].values[0]
        responses = row.response_list
        response_embeds = [spose_embeds[spose_embeds['concept'] == resp][spose_cols].values[0] for resp in responses]
        mean_response_embed = np.mean(response_embeds, axis=0)
        similarity_score = cosine_similarity(target_embed.reshape(1, -1), mean_response_embed.reshape(1, -1))[0][0]
        similarities.append(similarity_score)
    return similarities

def spose_permutation_test(df, spose_embeds, spose_cols, n_permutations=10):
    # Compute original similarities
    df['similarity_score'] = compute_similarities(df, spose_embeds, spose_cols)
    df_agg = df.groupby('uniqueID').agg({'similarity_score': 'mean', 'correct': 'mean'}).reset_index()
    original_r, _ = pearsonr(df_agg.similarity_score, df_agg.correct)
    
    # Permutation test
    permuted_rs = []
    for _ in range(n_permutations):
        df['similarity_score'] = compute_similarities(df, spose_embeds, spose_cols, random=True)
        df_agg = df.groupby('uniqueID').agg({'similarity_score': 'mean', 'correct': 'mean'}).reset_index()
        r, _ = pearsonr(df_agg.similarity_score, df_agg.correct)
        permuted_rs.append(r)
    
    p_value = np.mean(np.abs(permuted_rs) >= np.abs(original_r))
    
    return original_r, permuted_rs, p_value