import os


def load_image_pairs(path):
    img_pairs = []
    for path, _, fnames in os.walk(path):
        for fname in fnames:
            if not fname.endswith('.png'):
                continue
            source_img = os.path.join(path, fname)
            target_img = os.path.join(path, fname[:-4] + '.jpg')
            if os.path.exists(source_img) and os.path.exists(target_img):
                img_pairs.append((source_img, target_img))
    return img_pairs