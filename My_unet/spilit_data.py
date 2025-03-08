import os
import random

def check_matching_files(folder1, folder2):
    """
    检查两个文件夹中是否有相同的文件（不考虑扩展名）
    获取两个文件夹中的文件名（不含扩展名）
    """
    #os.path.splitext(f)返回一个元组 (root, ext)，其中 root 是文件名（不包含扩展名），而 ext 是文件的扩展名（包括点.）。
    files1 = set(os.path.splitext(f)[0] for f in os.listdir(folder1))
    files2 = set(os.path.splitext(f)[0] for f in os.listdir(folder2))

    # 找出匹配的文件
    #使用 intersection() 方法可以找到两个集合的交集，即在两个集合中都存在的元素。
    matching_files = list(files1.intersection(files2))
    return matching_files

def main(data_root="D:\\OneDrive\\桌面\\肝脏肿瘤语义分割\\含3D肝脏肿瘤数据集\\tmp"):
    # 设置随机种子以确保结果可复现
    random.seed(0)

    # 定义图像和掩码的路径
    images_path = os.path.join(data_root, "patient")
    masks_path = os.path.join(data_root, "tumor")

    # 确保路径存在
    assert os.path.exists(images_path), f"path '{images_path}' does not exist."
    assert os.path.exists(masks_path), f"path '{masks_path}' does not exist."

    # 先检查文件匹配情况
    matching_files = check_matching_files(images_path, masks_path)
    # 如果没有匹配文件，直接返回
    if not matching_files:
        print("No matching files found. Skipping dataset split.")
        return

    # 获取所有匹配图像的文件名（不含后缀）
    images = sorted(matching_files)  # 使用匹配的文件名
    num = len(images)

    # 随机打乱文件列表
    random.shuffle(images)

    # 划分数据集
    test_num = int(num * 0.1)  # 测试集占总数的10%
    val_num = int((num - test_num) * 0.1)  # 验证集占剩余数据的10%

    # 划分测试集
    test_images = images[:test_num]
    # 划分验证集
    val_images = images[test_num:test_num + val_num]
    # 划分训练集
    train_images = images[test_num + val_num:]

    # 写入文件
    #with open() 创建或打开，保证打开后自动关闭，w如果文件存在进行覆盖，追加使用a,
    with open(os.path.join(data_root, "train.txt"), 'w') as f:
        for image in train_images:
            f.write(f"{image}\n")

    with open(os.path.join(data_root, "val.txt"), 'w') as f:
        for image in val_images:
            f.write(f"{image}\n")

    with open(os.path.join(data_root, "test.txt"), 'w') as f:
        for image in test_images:
            f.write(f"{image}\n")

    print("Text files for dataset splits created successfully!")
    print(f"Total images: {num}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"Test images: {len(test_images)}")

if __name__ == '__main__':
    main()
