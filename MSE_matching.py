import torch
import cv2
import math
import csv
from pathlib import Path


# Порог RMSE: если ошибка меньше или равна этому значению, линия считается прямой
RMSE_THRESHOLD = 2.0

# Параметры робастной фильтрации: используются для удаления выбросных точек нейросети
ROBUST_ITERATIONS = 4
ROBUST_SIGMA = 2.5
ROBUST_MIN_GATE = 3.0

# Параметры памяти типа линии между кадрами
SHORT_LINE_Y_SPAN = 140
TYPE_SWITCH_FRAMES = 2

# Параметры сопоставления линий между соседними кадрами
MAX_MISSED_FRAMES = 3
MAX_ANGLE_DIFF_DEG = 12
MAX_X_DIFF_PX = 120
MAX_Y_MID_DIFF_PX = 220

# Параметры выбора осевой линии
MIN_AXIS_STABLE_FRAMES = 3
MIN_AXIS_Y_SPAN = 80

# Входные и выходные папки
IMAGES_FOLDER = Path("images")
LINES_FOLDER = Path("result_lines")
OUTPUT_FOLDER = Path("tracked_marked_images")
CSV_PATH = "tracking_result.csv"

OUTPUT_FOLDER.mkdir(exist_ok=True)


def read_lines_txt(path):
    # Читает txt-файл нейросети и преобразует каждую строку в набор точек линии
    lines = []

    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            nums = row.strip().split()

            if len(nums) < 4:
                continue

            nums = list(map(float, nums))

            if len(nums) % 2 != 0:
                nums = nums[:-1]

            points = torch.tensor(nums, dtype=torch.float64).view(-1, 2)
            lines.append(points)

    return lines


def fit_line_mnk(points, mask=None):
    # Строит прямую x = a * y + b методом наименьших квадратов
    if mask is None:
        used_points = points
    else:
        used_points = points[mask]

    x = used_points[:, 0]
    y = used_points[:, 1]

    A = torch.stack([y, torch.ones_like(y)], dim=1)
    solution = torch.linalg.lstsq(A, x).solution

    a = solution[0]
    b = solution[1]

    return a, b


def calc_distances(points, a, b):
    # Считает расстояния от точек линии до найденной прямой
    x = points[:, 0]
    y = points[:, 1]

    x_pred = a * y + b
    errors = torch.abs(x - x_pred) / torch.sqrt(a ** 2 + 1)

    return errors


def robust_fit_line(points):
    # Робастный МНК: сначала строит прямую, затем отбрасывает выбросные точки
    mask = torch.ones(len(points), dtype=torch.bool)

    for _ in range(ROBUST_ITERATIONS):
        if int(mask.sum()) < 2:
            break

        a, b = fit_line_mnk(points, mask)
        errors = calc_distances(points, a, b)

        used_errors = errors[mask]

        center = torch.median(used_errors)
        mad = torch.median(torch.abs(used_errors - center))

        robust_scale = 1.4826 * mad
        gate = max(ROBUST_MIN_GATE, float(ROBUST_SIGMA * robust_scale))

        new_mask = errors <= gate

        if torch.equal(new_mask, mask):
            break

        mask = new_mask

    if int(mask.sum()) < 2:
        mask = torch.ones(len(points), dtype=torch.bool)

    a, b = fit_line_mnk(points, mask)
    errors = calc_distances(points, a, b)

    inlier_errors = errors[mask]
    rmse = torch.sqrt(torch.mean(inlier_errors ** 2))

    return a, b, float(rmse), mask


def analyse_line(points, line_index):
    # Анализирует одну линию: RMSE, тип линии, длина по Y, угол и другие признаки
    a, b, rmse, inlier_mask = robust_fit_line(points)

    used_points = points[inlier_mask]

    y = used_points[:, 1]

    y_min = float(torch.min(y))
    y_max = float(torch.max(y))
    y_mid = (y_min + y_max) / 2
    y_span = y_max - y_min

    x_mid = float(a * y_mid + b)
    angle_deg = math.degrees(math.atan(float(a)))

    if rmse <= RMSE_THRESHOLD:
        raw_type = "STRAIGHT"
    else:
        raw_type = "CURVE"

    return {
        "line_index": line_index,
        "points": points,
        "inlier_mask": inlier_mask,
        "a": float(a),
        "b": float(b),
        "rmse": rmse,
        "raw_type": raw_type,
        "final_type": raw_type,
        "angle_deg": angle_deg,
        "x_mid": x_mid,
        "y_mid": y_mid,
        "y_min": y_min,
        "y_max": y_max,
        "y_span": y_span,
        "point_count": len(points),
        "inlier_count": int(inlier_mask.sum()),
        "track_id": None,
        "is_axis": False,
        "memory_reason": ""
    }


def analyse_file(path):
    # Анализирует все линии из одного txt-файла
    lines = read_lines_txt(path)
    results = []

    for i, points in enumerate(lines):
        results.append(analyse_line(points, i))

    return results


def get_file_key(file_path):
    # Получает имя кадра без служебных окончаний .onnxres.txt / .lines.txt
    name = file_path.name
    name = name.replace(".onnxres.txt", "")
    name = name.replace(".lines.txt", "")
    name = name.replace(".txt", "")
    return name


def find_image_by_key(key):
    # Ищет изображение, соответствующее текущему txt-файлу
    for ext in [".png", ".jpg", ".jpeg"]:
        image_path = IMAGES_FOLDER / f"{key}{ext}"

        if image_path.exists():
            return image_path

    return None


def frame_sort_key(path):
    # Сортировка кадров по числу в имени файла
    parts = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()

    if len(parts) > 0:
        return int(parts[-1])

    return path.name


def line_match_score(old_line, new_line):
    # Считает похожесть линии из прошлого кадра и линии из текущего кадра
    angle_diff = abs(old_line["angle_deg"] - new_line["angle_deg"])
    y_mid_diff = abs(old_line["y_mid"] - new_line["y_mid"])

    y1 = max(old_line["y_min"], new_line["y_min"])
    y2 = min(old_line["y_max"], new_line["y_max"])

    if y2 > y1:
        y_ref = (y1 + y2) / 2
    else:
        y_ref = new_line["y_mid"]

    old_x = old_line["a"] * y_ref + old_line["b"]
    new_x = new_line["a"] * y_ref + new_line["b"]

    x_diff = abs(old_x - new_x)

    if angle_diff > MAX_ANGLE_DIFF_DEG:
        return None

    if x_diff > MAX_X_DIFF_PX:
        return None

    if y_mid_diff > MAX_Y_MID_DIFF_PX:
        return None

    score = x_diff + angle_diff * 6 + y_mid_diff * 0.2

    return score


def update_track_type(track, line):
    # Обновляет стабильный тип линии с учетом памяти трека
    raw_type = line["raw_type"]
    stable_type = track["stable_type"]

    if stable_type == "CURVE" and raw_type == "STRAIGHT" and line["y_span"] < SHORT_LINE_Y_SPAN:
        line["final_type"] = "CURVE"
        line["memory_reason"] = "curve_memory_short_line"
        return

    if raw_type == stable_type:
        track["pending_type"] = raw_type
        track["pending_count"] = 0
        line["final_type"] = stable_type
        return

    if raw_type != track["pending_type"]:
        track["pending_type"] = raw_type
        track["pending_count"] = 1
    else:
        track["pending_count"] += 1

    if track["pending_count"] >= TYPE_SWITCH_FRAMES:
        track["stable_type"] = raw_type
        track["pending_count"] = 0

    line["final_type"] = track["stable_type"]

    if line["final_type"] != raw_type:
        line["memory_reason"] = "type_memory"


def draw_lines(image, lines, frame_name):
    # Рисует линии на изображении: зеленая — осевая, синяя — прямая, красная — кривая
    height, width = image.shape[:2]

    for line in lines:
        if line["is_axis"]:
            color = (0, 255, 0)
            thickness = 5
            label_type = "AXIS " + line["final_type"]
        elif line["final_type"] == "CURVE":
            color = (0, 0, 255)
            thickness = 3
            label_type = "CURVE"
        else:
            color = (255, 0, 0)
            thickness = 3
            label_type = "STRAIGHT"

        points = line["points"].numpy()
        draw_points = []

        for x, y in points:
            x = int(round(x))
            y = int(round(y))

            if 0 <= x < width and 0 <= y < height:
                draw_points.append((x, y))

        if len(draw_points) == 0:
            continue

        for i in range(len(draw_points) - 1):
            cv2.line(image, draw_points[i], draw_points[i + 1], color, thickness)

        for point in draw_points:
            cv2.circle(image, point, 3, color, -1)

        label = (
            f"line {line['line_index']} {label_type} "
            f"track={line['track_id']} RMSE={line['rmse']:.2f} "
            f"inl={line['inlier_count']}/{line['point_count']}"
        )

        if line["memory_reason"] != "":
            label += " memory"

        text_x, text_y = draw_points[len(draw_points) // 2]

        cv2.putText(
            image,
            label,
            (text_x + 10, text_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2
        )

    cv2.putText(
        image,
        frame_name,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2
    )

    return image


# Состояние трекинга между кадрами
tracks = []
next_track_id = 0
main_axis_track_id = None
csv_rows = []

txt_files = sorted(LINES_FOLDER.glob("*.txt"), key=frame_sort_key)


for txt_path in txt_files:
    key = get_file_key(txt_path)
    lines = analyse_file(txt_path)

    used_lines = set()

    # Сопоставляем найденные линии с уже существующими треками
    for track in tracks:
        best_i = None
        best_score = None

        for i, line in enumerate(lines):
            if i in used_lines:
                continue

            score = line_match_score(track["last_line"], line)

            if score is None:
                continue

            if best_score is None or score < best_score:
                best_score = score
                best_i = i

        if best_i is not None:
            line = lines[best_i]

            track["last_line"] = line
            track["hits"] += 1
            track["consecutive_hits"] += 1
            track["missed"] = 0

            line["track_id"] = track["id"]

            update_track_type(track, line)

            used_lines.add(best_i)
        else:
            track["missed"] += 1
            track["consecutive_hits"] = 0

    # Для новых линий создаем новые треки
    for i, line in enumerate(lines):
        if i in used_lines:
            continue

        track = {
            "id": next_track_id,
            "last_line": line,
            "hits": 1,
            "consecutive_hits": 1,
            "missed": 0,
            "stable_type": line["raw_type"],
            "pending_type": line["raw_type"],
            "pending_count": 0
        }

        line["track_id"] = next_track_id
        line["final_type"] = line["raw_type"]

        tracks.append(track)
        next_track_id += 1

    tracks = [
        track for track in tracks
        if track["missed"] <= MAX_MISSED_FRAMES
    ]

    track_by_id = {track["id"]: track for track in tracks}

    axis_line = None

    # Если осевая линия уже была найдена раньше, стараемся продолжать тот же трек
    if main_axis_track_id is not None:
        for line in lines:
            if (
                line["track_id"] == main_axis_track_id
                and line["y_span"] >= MIN_AXIS_Y_SPAN
            ):
                axis_line = line
                break

    # Если осевая не найдена по старому треку, выбираем наиболее стабильную линию
    if axis_line is None:
        candidates = []

        for line in lines:
            if line["y_span"] < MIN_AXIS_Y_SPAN:
                continue

            track = track_by_id.get(line["track_id"])

            if track is None:
                continue

            if track["consecutive_hits"] < MIN_AXIS_STABLE_FRAMES:
                continue

            candidates.append(line)

        if len(candidates) > 0:
            axis_line = max(
                candidates,
                key=lambda line: (
                    track_by_id[line["track_id"]]["consecutive_hits"],
                    line["y_span"],
                    line["inlier_count"],
                    -line["rmse"]
                )
            )

            main_axis_track_id = axis_line["track_id"]

    if axis_line is not None:
        axis_line["is_axis"] = True

    print("Кадр:", key)

    if axis_line is None:
        print("Осевая линия: не выбрана")
    else:
        print(
            "Осевая линия:",
            "line", axis_line["line_index"],
            "| type", axis_line["final_type"],
            "| track", axis_line["track_id"],
            "| RMSE", round(axis_line["rmse"], 3)
        )

    for line in lines:
        print(
            "line", line["line_index"],
            "| raw:", line["raw_type"],
            "| final:", line["final_type"],
            "| RMSE:", round(line["rmse"], 3),
            "| inliers:", line["inlier_count"], "/", line["point_count"],
            "| y_span:", round(line["y_span"], 1),
            "| track:", line["track_id"],
            "|", line["memory_reason"]
        )

        csv_rows.append({
            "frame": key,
            "line_index": line["line_index"],
            "track_id": line["track_id"],
            "raw_type": line["raw_type"],
            "final_type": line["final_type"],
            "rmse": line["rmse"],
            "angle_deg": line["angle_deg"],
            "x_mid": line["x_mid"],
            "y_mid": line["y_mid"],
            "y_span": line["y_span"],
            "point_count": line["point_count"],
            "inlier_count": line["inlier_count"],
            "is_axis": line["is_axis"],
            "memory_reason": line["memory_reason"]
        })

    image_path = find_image_by_key(key)

    if image_path is not None:
        image = cv2.imread(str(image_path))

        if image is not None:
            marked = draw_lines(image, lines, key)
            out_path = OUTPUT_FOLDER / f"{key}_tracked.png"
            cv2.imwrite(str(out_path), marked)

    print("-" * 40)


# Сохраняем таблицу с результатами классификации и трекинга
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "frame",
        "line_index",
        "track_id",
        "raw_type",
        "final_type",
        "rmse",
        "angle_deg",
        "x_mid",
        "y_mid",
        "y_span",
        "point_count",
        "inlier_count",
        "is_axis",
        "memory_reason"
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)


print("Готово")
print("Размеченные изображения:", OUTPUT_FOLDER)
print("CSV:", CSV_PATH)