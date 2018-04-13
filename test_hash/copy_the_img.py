import shutil

import os


def copy_img(text_dirs):
    target_dir = os.path.dirname(text_dirs)
    with open(text_dirs, "r") as f:
        contents = f.readlines()
        lines = len(contents)
    for i in range(0, lines, 3):
        img_dir = contents[i].strip('\n').split('错误路径:')[1]
        doc = img_dir.split('\\')[-3]
        new_dir = os.path.join(target_dir, doc)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        shutil.copy(img_dir, new_dir)


if __name__ == '__main__':
    copy_img("C:\\Users\\fdgfd\\Desktop\\杂草\\准确率\\4conv_128_128\\4conv_128_128.txt")
