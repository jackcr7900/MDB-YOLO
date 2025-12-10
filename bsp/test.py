from ultralytics import YOLO
import os
import csv
from datetime import datetime

# =========================
# 配置
# =========================
model = YOLO("model_20250930/MDB-YOLO.pt")  # 加载模型
frame_folder = r"E:\BaiduNetdiskDownload\Network-YOLOv\ultralytics-8.2.0_test\bsp\frame"
output_folder = r"E:\BaiduNetdiskDownload\Network-YOLOv\ultralytics-8.2.0_test\bsp\val_frame"
os.makedirs(output_folder, exist_ok=True)

# 支持的图片格式
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [f for f in os.listdir(frame_folder)
               if os.path.splitext(f)[1].lower() in image_extensions]

if not image_files:
    raise RuntimeError("❌ 没有找到图片，请检查 frame_folder 路径")

# =========================
# 预热
# =========================
print("⚡ 模型预热中...\n")
_ = model([os.path.join(frame_folder, image_files[0])])  # 不计入统计
print("✅ 预热完成，开始正式检测...\n")

time_records = []  # 存储每张图的时间信息

print(f"找到 {len(image_files)-1} 张图片 (去掉预热)，开始检测...\n")
print(f"{'序号':<6} {'文件名':<20} {'预处理(ms)':<12} {'推理(ms)':<12} {'后处理(ms)':<12} {'总时间(ms)':<12}")
print("=" * 85)

# =========================
# 正式推理（从第2张开始）
# =========================
for idx, image_file in enumerate(image_files[1:], 1):
    image_path = os.path.join(frame_folder, image_file)

    results = model([image_path])

    for result in results:
        speed = result.speed  # dict: preprocess, inference, postprocess
        preprocess_time = speed['preprocess']
        inference_time = speed['inference']
        postprocess_time = speed['postprocess']
        total_time = preprocess_time + inference_time + postprocess_time

        num_detections = len(result.boxes) if result.boxes is not None else 0

        time_records.append({
            '序号': idx,
            '文件名': image_file,
            '预处理时间(ms)': round(preprocess_time, 2),
            '推理时间(ms)': round(inference_time, 2),
            '后处理时间(ms)': round(postprocess_time, 2),
            '总时间(ms)': round(total_time, 2),
            '检测目标数': num_detections
        })

        # 保存检测结果图片
        filename_without_ext = os.path.splitext(image_file)[0]
        save_path = os.path.join(output_folder, f"{filename_without_ext}.jpg")
        result.save(filename=save_path)

        # 打印日志
        print(f"{idx:<6} {image_file:<20} {preprocess_time:<12.2f} "
              f"{inference_time:<12.2f} {postprocess_time:<12.2f} {total_time:<12.2f}")

# =========================
# 统计平均值 + FPS
# =========================
avg_preprocess = sum(r['预处理时间(ms)'] for r in time_records) / len(time_records)
avg_inference = sum(r['推理时间(ms)'] for r in time_records) / len(time_records)
avg_postprocess = sum(r['后处理时间(ms)'] for r in time_records) / len(time_records)
avg_total = sum(r['总时间(ms)'] for r in time_records) / len(time_records)

fps_inference = 1000.0 / avg_inference
fps_total = 1000.0 / avg_total

print("=" * 85)
print(f"{'平均':<6} {'':<20} {avg_preprocess:<12.2f} {avg_inference:<12.2f} "
      f"{avg_postprocess:<12.2f} {avg_total:<12.2f}")

print("\n===== 性能评估 =====")
print(f"仅推理 FPS: {fps_inference:.2f}")
print(f"端到端 FPS (预处理+推理+后处理): {fps_total:.2f}")

# =========================
# 写入 CSV
# =========================
csv_filename = os.path.join(output_folder, f"inference_time_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
    fieldnames = ['序号', '文件名', '预处理时间(ms)', '推理时间(ms)', '后处理时间(ms)', '总时间(ms)', '检测目标数']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(time_records)
    # 追加平均值
    writer.writerow({
        '序号': '',
        '文件名': '平均值',
        '预处理时间(ms)': round(avg_preprocess, 2),
        '推理时间(ms)': round(avg_inference, 2),
        '后处理时间(ms)': round(avg_postprocess, 2),
        '总时间(ms)': round(avg_total, 2),
        '检测目标数': round(sum(r['检测目标数'] for r in time_records) / len(time_records), 1)
    })

print(f"\n✓ 所有图片检测完成！")
print(f"✓ 检测结果已保存到: {output_folder}")
print(f"✓ 时间记录已保存到: {csv_filename}")


# # Run batched inference on a list of images
# results = model(["frame/11.jpg"])  # return a list of Results objects
# # 绮惧害瓒婇珮锛屾�€娴嬫椂闂磋秺闀�
#
# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     # result.show()
#     # result.plot(conf=False, labels=False)
#     result.show(conf=True, labels=True)  # display to screen
#     #result.save(filename="images/resources/result.jpg")  # save to disk
