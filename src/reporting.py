import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS_BG = (255, 255, 255)
TEXT = (38, 50, 56)
MUTED = (84, 110, 122)
BLUE = (37, 92, 122)
RED = (209, 73, 91)
GRID = (214, 222, 230)
LIGHT_BLUE = (235, 247, 250)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_line_chart(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    epochs: list[int],
    train_values: list[float],
    val_values: list[float],
    y_min: float,
    y_max: float,
    y_label_format,
) -> None:
    title_font = _font(22, bold=True)
    label_font = _font(14)
    small_font = _font(12)
    draw.text((x, y), title, fill=TEXT, font=title_font)
    chart_x = x + 70
    chart_y = y + 58
    chart_w = width - 100
    chart_h = height - 100

    for i in range(5):
        yy = chart_y + chart_h - int(chart_h * i / 4)
        draw.line((chart_x, yy, chart_x + chart_w, yy), fill=GRID, width=1)
        value = y_min + (y_max - y_min) * i / 4
        label = y_label_format(value)
        tw, th = _text_size(draw, label, small_font)
        draw.text((chart_x - tw - 10, yy - th // 2), label, fill=MUTED, font=small_font)

    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill=TEXT, width=2)
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=TEXT, width=2)

    def point(index: int, value: float) -> tuple[int, int]:
        x_pos = chart_x + int(chart_w * index / max(1, len(epochs) - 1))
        normalized = (value - y_min) / (y_max - y_min)
        y_pos = chart_y + chart_h - int(chart_h * normalized)
        return x_pos, max(chart_y, min(chart_y + chart_h, y_pos))

    for values, color in [(train_values, BLUE), (val_values, RED)]:
        points = [point(i, value) for i, value in enumerate(values)]
        for start, end in zip(points, points[1:]):
            draw.line((start, end), fill=color, width=3)
        for px, py in points[:: max(1, len(points) // 6)]:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline=color, width=2)

    for epoch in [epochs[0], 5, 10, 15, epochs[-1]]:
        if epoch in epochs:
            index = epochs.index(epoch)
            px, _ = point(index, y_min)
            label = str(epoch)
            tw, _ = _text_size(draw, label, small_font)
            draw.text((px - tw // 2, chart_y + chart_h + 12), label, fill=MUTED, font=small_font)

    legend_x = chart_x + chart_w - 230
    legend_y = y + 18
    draw.line((legend_x, legend_y + 8, legend_x + 35, legend_y + 8), fill=BLUE, width=4)
    draw.text((legend_x + 45, legend_y), "Train", fill=TEXT, font=label_font)
    draw.line((legend_x + 120, legend_y + 8, legend_x + 155, legend_y + 8), fill=RED, width=4)
    draw.text((legend_x + 165, legend_y), "Validation", fill=TEXT, font=label_font)


def load_training_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_accuracy": float(row["train_accuracy"]),
                "val_loss": float(row["val_loss"]),
                "val_accuracy": float(row["val_accuracy"]),
            }
            for row in reader
        ]


def save_learning_curves(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        raise ValueError("Training history is empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), CANVAS_BG)
    draw = ImageDraw.Draw(image)

    title_font = _font(32, bold=True)
    body_font = _font(16)
    draw.text((50, 30), "EuroSAT CNN Learning Curves", fill=TEXT, font=title_font)
    draw.text(
        (50, 75),
        "Training and validation metrics from reports/training_history.csv",
        fill=MUTED,
        font=body_font,
    )

    epochs = [int(row["epoch"]) for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    _draw_line_chart(
        draw,
        50,
        125,
        1300,
        340,
        "Accuracy",
        epochs,
        train_acc,
        val_acc,
        min(train_acc + val_acc) - 0.03,
        max(train_acc + val_acc) + 0.03,
        lambda value: f"{value * 100:.0f}%",
    )
    _draw_line_chart(
        draw,
        50,
        510,
        1300,
        340,
        "Loss",
        epochs,
        train_loss,
        val_loss,
        max(0.0, min(train_loss + val_loss) - 0.1),
        max(train_loss + val_loss) + 0.1,
        lambda value: f"{value:.2f}",
    )
    image.save(output_path)


def save_confusion_matrix_image(
    confusion: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    short_names = [
        "Annual",
        "Forest",
        "HerbVeg",
        "Highway",
        "Industrial",
        "Pasture",
        "PermCrop",
        "Resident",
        "River",
        "SeaLake",
    ]
    if len(short_names) != len(class_names):
        short_names = [name[:8] for name in class_names]

    cell = 78
    left = 170
    top = 210
    width = left + cell * len(class_names) + 70
    height = top + cell * len(class_names) + 150
    image = Image.new("RGB", (width, height), CANVAS_BG)
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    label_font = _font(15)
    small_font = _font(13)
    number_font = _font(14, bold=True)

    draw.text((40, 30), "Validation Confusion Matrix", fill=TEXT, font=title_font)
    draw.text(
        (40, 78),
        "Rows are true classes. Columns are predicted classes. Dark diagonal cells are correct predictions.",
        fill=MUTED,
        font=label_font,
    )

    max_value = max(max(row) for row in confusion)
    for index, label in enumerate(short_names):
        x = left + index * cell + cell // 2
        label_image = Image.new("RGBA", (120, 30), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((0, 0), label, fill=TEXT, font=small_font)
        rotated = label_image.rotate(45, expand=True)
        image.paste(rotated, (x - 35, top - 105), rotated)

        y = top + index * cell + cell // 2
        tw, th = _text_size(draw, label, small_font)
        draw.text((left - tw - 16, y - th // 2), label, fill=TEXT, font=small_font)

    for row_index, row in enumerate(confusion):
        for col_index, value in enumerate(row):
            x = left + col_index * cell
            y = top + row_index * cell
            intensity = value / max_value if max_value else 0
            if row_index == col_index:
                color = BLUE
                text_color = (255, 255, 255)
            else:
                shade = 248 - int(95 * intensity)
                color = (shade, min(255, shade + 6), 255)
                text_color = TEXT
            draw.rectangle((x, y, x + cell, y + cell), fill=color, outline=(255, 255, 255), width=2)
            label = str(value)
            tw, th = _text_size(draw, label, number_font)
            draw.text((x + cell // 2 - tw // 2, y + cell // 2 - th // 2), label, fill=text_color, font=number_font)

    draw.text((left + cell * len(class_names) // 2 - 45, height - 60), "Predicted class", fill=MUTED, font=label_font)
    draw.text((40, top + cell * len(class_names) // 2), "True class", fill=MUTED, font=label_font)
    image.save(output_path)


def save_per_class_accuracy_csv(
    class_names: list[str],
    per_class_accuracy: list[float],
    confusion: list[list[int]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "accuracy", "correct", "total"],
        )
        writer.writeheader()
        for index, class_name in enumerate(class_names):
            correct = confusion[index][index]
            total = sum(confusion[index])
            writer.writerow(
                {
                    "class_name": class_name,
                    "accuracy": f"{per_class_accuracy[index]:.6f}",
                    "correct": correct,
                    "total": total,
                }
            )
