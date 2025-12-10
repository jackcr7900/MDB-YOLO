from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.cm as cm


def create_gradcam_heatmap(model, image_path, output_path):
    """
    使用Grad-CAM技术生成热力图并保存（只保存热力图部分）
    """
    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 使用YOLO进行推理并获取特征图
    results = model(image_path, verbose=False)

    # 获取检测结果
    result = results[0]

    # 创建热力图
    # 基于检测框创建热力图
    if result.boxes is not None and len(result.boxes) > 0:
        # 获取图像尺寸
        h, w = img_rgb.shape[:2]

        # 创建空白热力图
        heatmap = np.zeros((h, w), dtype=np.float32)

        # 为每个检测框添加热力图区域
        for box in result.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()

            # 转换为整数坐标
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # 在热力图中添加高斯分布
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            sigma = max((x2 - x1) // 4, (y2 - y1) // 4, 10)

            # 创建高斯核
            y, x = np.ogrid[:h, :w]
            gaussian = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
            gaussian *= conf  # 根据置信度调整强度

            # 累加到热力图
            heatmap = np.maximum(heatmap, gaussian)

        # 归一化热力图
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # 应用颜色映射
        heatmap_colored = cm.jet(heatmap)[:, :, :3]

        # 叠加到原图
        overlay = 0.6 * heatmap_colored + 0.4 * (img_rgb / 255.0)

        # 转换为 0-255 范围并保存
        overlay_img = (overlay * 255).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, overlay_bgr)
    else:
        # 没有检测到目标时保存原图
        cv2.imwrite(output_path, img)


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    # 1️⃣ 配置
    model = YOLO("model_20250930/MDB-YOLO.pt")
    frame_folder = r"E:\BaiduNetdiskDownload\Network-YOLOv\ultralytics-8.2.0_test\\bsp\frame"
    output_folder = r"E:\BaiduNetdiskDownload\Network-YOLOv\ultralytics-8.2.0_test\bsp\val_frame"

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 2️⃣ 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(frame_folder)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print(f"❌ 没有找到图片，请检查 frame_folder 路径: {frame_folder}")
        exit(1)

    print(f"🚀 找到 {len(image_files)} 张图片，开始生成Grad-CAM热力图...\n")
    print(f"{'序号':<6} {'文件名':<20} {'状态':<30}")
    print("=" * 60)

    # 3️⃣ 遍历所有图片
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(frame_folder, image_file)
        filename_without_ext = os.path.splitext(image_file)[0]

        # 生成Grad-CAM热力图
        output_path = os.path.join(output_folder, f"{filename_without_ext}.jpg")
        create_gradcam_heatmap(model, image_path, output_path)

        print(f"{idx:<6} {image_file:<20} {'✅ 完成':<30}")

    print("=" * 60)
    print(f"\n🎉 所有热力图生成完成！")
    print(f"📁 结果保存在: {output_folder}")