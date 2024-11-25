import albumentations as A

def augment1():

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.5
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
    ])
    return transform
