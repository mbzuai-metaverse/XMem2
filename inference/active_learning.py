from sklearn.cluster import KMeans
import pandas as pd
from PIL import Image
from torchvision.transforms import ColorJitter, Grayscale, RandomPosterize, RandomAdjustSharpness, ToTensor

def select_n_frame_candidates(preds_df: pd.DataFrame, uncertainty_name: str, n=5):
    df = preds_df

    df.reset_index(drop=False, inplace=True)

    # max_frame = df['frame'].max()
    # max_entropy = df['entropy'].max()
    
    df = df[df['mask_provided'] == False]  # removing frames with masks
    df = df[df[uncertainty_name] >= df[uncertainty_name].median()] # removing low entropy parts
    
    df_backup = df.copy()
    
    df['index'] = df['index'] / df['index'].max() # scale to 0..1
    # df['entropy'] = df['entropy'] / df['entropy'].max() # scale to 0..1
    
    X = df[['index', uncertainty_name]].to_numpy()

    clusterer = KMeans(n_clusters=n)
    
    labels = clusterer.fit_predict(X)

    clusters = df_backup.groupby(labels)
    
    candidates = []

    for g, cluster in clusters:
        if g == -1:
            continue
        
        max_entropy_idx = cluster[uncertainty_name].argmax()

        res = cluster.iloc[max_entropy_idx]

        candidates.append(res)

    return candidates

def select_most_uncertain_frame(preds_df: pd.DataFrame, uncertainty_name: str):
    preds_df.reset_index(drop=False, inplace=True)
    return preds_df.iloc[preds_df[uncertainty_name].argmax()]


def get_determenistic_augmentations():
    # TODO: maybe add GaussianBlur?

    bright = ColorJitter(brightness=(1.5, 1.5))
    dark = ColorJitter(brightness=(0.5, 0.5))
    gray = Grayscale(num_output_channels=3)
    reduce_bits = RandomPosterize(bits=3, p=1)
    sharp = RandomAdjustSharpness(sharpness_factor=16, p=1)
    
    return [bright, dark, gray, reduce_bits, sharp]

def apply_aug(img_path, out_path):
    img = Image.open(img_path)
    
    bright, dark, gray, reduce_bits, sharp = get_determenistic_augmentations()
    
    img_augged = sharp(img)
    
    img_augged.save(out_path)
    

if __name__ == '__main__':
    img_in = '/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/JPEGImages/frame_000001.PNG'
    img_out = 'test_aug.png'
    
    apply_aug(img_in, img_out)
    