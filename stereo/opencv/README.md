# 双目 ChArUco 标定（OpenCV）

`capture_stereo_pairs.py` 迁自 `rk3588_cross_camera_tracking/tools/stereo_calibration/`，`stereo_calibrate.py` 迁自 `yolov8_3d_coord/tools/`。工具独立运行，不依赖原项目目录。

## 环境与路径

在 coord 根目录执行 `python -m pip install -r requirements.txt`，使用带 GUI 与 ArUco 模块的 `opencv-contrib-python`。应避免在同一环境中混装多个提供 `cv2` 的 OpenCV 包。

所有命令参数及配置中的相对路径均相对于 **coord 根目录**，绝对路径也受支持。可以直接指定原项目素材目录，无需复制图片。

## 1. 拍摄素材

```powershell
python stereo/opencv/capture_stereo_pairs.py --left 0 --right 1 --width 1920 --height 1080 --fps 30 --output-dir data/calibration
```

- `Space / S` 保存一对原图，`Q / Esc` 退出。
- 输出 `data/calibration/left/000001.png` 和 `right/000001.png`，同名表示同一对。
- 已有成对素材会继续编号，不覆盖已有图片；左右不配对时停止。
- Windows 默认 DirectShow、MJPG，可通过 `--backend`、`--fourcc` 调整。
- 先 grab 再 retrieve 减小时间差，属于软件近同步；拍摄时保持板静止。
- 实际分辨率与请求不符时停止。预览缩放不影响保存的原始图像。

建议拍摄 20–30 对不同距离、位置、倾角的清晰图片；至少需要 10 对有效图片。左右相机安装状态、焦距、对焦和分辨率应与实际使用时一致。

## 2. 核对配置

编辑同目录 `stereo_calibrate_charuco.yml`：

| 配置项 | 默认值 |
| --- | --- |
| `left_dir` / `right_dir` | `data/calibration/left` / `data/calibration/right` |
| `left_intrinsics_csv` | `csv/cd_left_1920x1080_intrinsics.csv` |
| `right_intrinsics_csv` | `csv/cd_right_1920x1080_intrinsics.csv` |
| `cols` / `rows` | 内部角点 10×7，即棋盘格 11×8 |
| `square_size_m` | 0.068 米（实测方格边长） |
| `marker_size_m` | 0.0535 米（实测标记边长） |
| `aruco_dictionary` | `DICT_6X6_50` |
| `output` | 根目录 `stereo.yml` |

内参必须对应实际左右相机、镜头状态及素材分辨率；程序不自动缩放内参。默认的两份 CSV 仅是现有文件配置，并不代表适用于任意相机。

两个内参路径都填写时，固定内参求双目外参。两个都设为 `""` 时，先从素材估计各自内参，再求双目外参；不能只填写一侧。

CSV 为 5×3：前三行 K，第四行 `k1,k2,k3`，第五行 `p1,p2,0`。文件名含 `MATLAB` 时转置 K，与 `world/camera.py` 一致。单目 OpenCV 导出的十位小数 CSV 可直接读取，9 行 `world_params.csv` 则应先提取前 5 行。

## 3. 联合标定

```powershell
python stereo/opencv/stereo_calibrate.py --config stereo/opencv/stereo_calibrate_charuco.yml
```

省略 `--config` 会使用同目录默认配置。支持命令行覆盖 `--left-dir`、`--right-dir`、`--left-intrinsics-csv`、`--right-intrinsics-csv`、`--output`、`--preview-mode`、`--preview-delay-ms`。

左右图片文件名集合必须相同，支持 PNG/JPG/JPEG/BMP/TIF/TIFF。至少 6 个共同角点的图片对才能参与计算，图像分辨率需一致。

| 审核按键 | 操作 |
| --- | --- |
| `N / Space / Enter` | 接受有效图片对并继续，无效对自动跳过 |
| `B` | 回看上一对 |
| `D` | 从磁盘删除当前左右两张原图 |
| `1 / 2 / 3` | 左图 / 右图 / 双图 |
| `Z` | 切换板区域放大与完整画面 |
| `F` | 全屏 |
| `Q / Esc` | 中止，保留原输出文件 |

无窗口处理：

```powershell
python stereo/opencv/stereo_calibrate.py --preview-mode off
```

自动预览：`--preview-mode time --preview-delay-ms 500`。标定成功后覆盖配置的输出文件。

## 4. 输出与对接

`stereo.yml` 保持两个原项目使用的 OpenCV FileStorage 字段：

```text
image_width, image_height
camera_matrix_left, distortion_left
camera_matrix_right, distortion_right
rotation_left_to_right, translation_left_to_right
rms_stereo, valid_pair_count
rms_left, rms_right  # 仅自动估计内参时存在
```

方向为 `X_right = R * X_left + T`，平移单位为米，基线为 `norm(T)`。控制台打印基线、有效对数和 RMS；固定 CSV 内参时未重算单目 RMS，所以控制台对应项为 `nan`，YAML 不写入这两项。

结果可用于 `yolov8_3d_coord`；用于 `rk3588_cross_camera_tracking` 时设置 `calibration_unit: m`。左右顺序与图像尺寸必须一致。这里不生成世界坐标系外参；该步骤使用 `world/get_params.py`。

## 验证

```powershell
python -m unittest discover -s stereo/opencv -v
```

测试使用临时生成的双目 ChArUco 图片，验证固定/自动内参、R/T、YAML 字段、CSV、配对和中止行为，无需相机或 GUI。USB 采集、人工审核及 MATLAB 会话需在对应环境中验证。
