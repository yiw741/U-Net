from My_unet import UNet
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np

# 调色板定义
palette = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128), (128, 64, 12)
]

def save_images(image, mask, output_path, image_file, palette, num_classes):
    """
    保存分割结果图像
    """
    os.makedirs(output_path, exist_ok=True)
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = cam_mask(mask, palette, num_classes)
    colorized_mask.save(os.path.join(output_path, f"{image_file}_segmentation.png"))

def cam_mask(mask, palette, n):
    """
    将分割掩码转换为彩色图像
    """
    seg_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for c in range(n):
        seg_img[:, :, 0] += ((mask == c) * palette[c][0]).astype('uint8')
        seg_img[:, :, 1] += ((mask == c) * palette[c][1]).astype('uint8')
        seg_img[:, :, 2] += ((mask == c) * palette[c][2]).astype('uint8')
    return Image.fromarray(seg_img)

def predict_single_image(
        img_path,
        weight_path,
        output_path,
        num_classes=2,
        model_class=UNet
):
    """
    对单张图像进行语义分割预测
    """
    os.makedirs(output_path, exist_ok=True)

    # 检查图像文件是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # 简化预处理，只转换为张量
    transform = transforms.Compose([
        transforms.ToTensor()  # 仅转换为张量，不进行归一化
    ])

    # 设备选择
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        # 详细的模型加载和调试日志
        print(f"Loading model from {weight_path}")
        print(f"Using device: {device}")

        # 初始化模型
        model = model_class(out_channel=num_classes)

        # 加载权重的更robust方法
        try:
            checkpoint = torch.load(weight_path, map_location=device)

            # 打印检查点信息
            if isinstance(checkpoint, dict):
                print("Checkpoint keys:", checkpoint.keys())

            # 多种权重加载策略
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise

        # 移动模型到设备
        model.to(device)
        model.eval()

        # 图像处理和预测
        with torch.no_grad():
            # 打开图像
            image = Image.open(img_path).convert('RGB')
            print(f"Original image size: {image.size}")

            # 图像转换
            input_tensor = transform(image).unsqueeze(0).to(device)

            # 调试输入张量
            print("Input tensor shape:", input_tensor.shape)
            print("Input tensor range:",
                  input_tensor.min().item(),
                  input_tensor.max().item())

            # 模型预测
            prediction = model(input_tensor)

            # 调试预测结果
            print("Prediction shape:", prediction.shape)
            print("Prediction value range:",
                  prediction.min().item(),
                  prediction.max().item())

            # 处理预测结果
            prediction = prediction.squeeze(0).cpu()
            prediction = F.softmax(prediction, dim=0).argmax(0).numpy()

            # 保存分割结果
            save_images(
                image,
                prediction,
                output_path,
                img_path,
                palette,
                num_classes
            )

        # 分析预测结果
        unique_pixels = np.unique(prediction)
        unique_counts = np.unique(prediction, return_counts=True)

        print("\n不同像素值:")
        print(unique_pixels)
        print("\n像素值分布:")
        for val, count in zip(unique_counts[0], unique_counts[1]):
            print(f"类别 {val}: {count} 个像素")

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        import traceback
        traceback.print_exc()

def predict_multiple_images(
        img_dir,
        weight_path,
        output_path,
        num_classes=2,
        model_class=UNet
):
    """
    批量预测图像
    """
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(img_dir):
        if filename.endswith(('.tif', '.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_dir, filename)
            try:
                predict_single_image(
                    img_path,
                    weight_path,
                    output_path,
                    num_classes,
                    model_class
                )
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 主程序入口
if __name__ == "__main__":
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 模型权重路径
    weight_path = os.path.join(current_dir, 'weights', 'best_model.pth')

    # 单张图像预测示例
    img_file = os.path.join(current_dir, '2007_000027.jpg')
    output_path = os.path.join(current_dir, 'weights', 'predictions')

    # 单张图像预测
    predict_single_image(
        img_file,
        weight_path,
        output_path,
        num_classes=21
    )

    # 批量图像预测（可选）
    # img_dir = os.path.join(current_dir, 'test_images')
    # predict_multiple_images(
    #     img_dir,
    #     weight_path,
    #     output_path,
    #     num_classes=2
    # )