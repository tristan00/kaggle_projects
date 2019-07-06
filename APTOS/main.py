import glob
from sklearn.model_selection import train_test_split
from PIL import Image
path = r'C:\Users\trist\Downloads\aptos2019-blindness-detection'


def augment_image(img):
    return img


def image_gen(image_paths, augmentation_allowed):
    for i in image_paths:
        img = Image.open(i)

        if augmentation_allowed:
            img = augment_image(img)

        yield img


def get_images():
    files = glob.glob('{0}.png'.format(path))
    train_files, val_files = train_test_split(files, random_state=1)
    return image_gen(train_files, True), image_gen(train_files, False)




if __name__ == '__main__':
    train_gen, val_gen = get_data()