from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Укажите путь к нужному изображению вручную.
IMAGE_PATH = PROJECT_ROOT / "data" / "lines" / "a01" / "a01-000u" / "a01-000u-00.png"

# Если указать директорию, итоговый файл будет сохранён в ней.
OUTPUT_PATH: Path | None = PROJECT_ROOT / "data" / "marked"

DRAW_LINE_BOXES = True
DRAW_WORD_BOXES = True
DRAW_COMPONENT_BOXES = True
DRAW_LABELS = False
LINE_WIDTH = 2

COLORS = {
    "target": (255, 196, 0),
    "line": (255, 80, 80),
    "word": (80, 220, 120),
    "cmp": (80, 160, 255),
}


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def translated(self, dx: int, dy: int) -> "Box":
        return Box(
            left=self.left + dx,
            top=self.top + dy,
            right=self.right + dx,
            bottom=self.bottom + dy,
        )

    def scaled(self, scale_x: float, scale_y: float) -> "Box":
        return Box(
            left=round(self.left * scale_x),
            top=round(self.top * scale_y),
            right=round(self.right * scale_x),
            bottom=round(self.bottom * scale_y),
        )


@dataclass(frozen=True)
class ImageTarget:
    kind: str
    form_id: str
    entity_id: str
    component_index: int | None = None


@dataclass(frozen=True)
class Annotation:
    box: Box
    color: tuple[int, int, int]
    label: str | None = None


def parse_image_target(image_path: Path) -> ImageTarget:
    parts = image_path.stem.split("-")
    if len(parts) < 2:
        raise ValueError(
            "Имя файла должно содержать как минимум form id, например a01-003.png."
        )

    form_id = "-".join(parts[:2])
    if len(parts) == 2:
        return ImageTarget(kind="form", form_id=form_id, entity_id=form_id)
    if len(parts) == 3:
        return ImageTarget(
            kind="line",
            form_id=form_id,
            entity_id="-".join(parts[:3]),
        )
    if len(parts) == 4:
        return ImageTarget(
            kind="word",
            form_id=form_id,
            entity_id="-".join(parts[:4]),
        )

    try:
        component_index = int(parts[4])
    except ValueError as exc:
        raise ValueError(
            "Для изображения символа ожидается имя вида a01-003-00-04-02.png."
        ) from exc

    return ImageTarget(
        kind="component",
        form_id=form_id,
        entity_id="-".join(parts[:4]),
        component_index=component_index,
    )


def find_xml_path(form_id: str) -> Path:
    direct_match = PROJECT_ROOT / "data" / "xml" / f"{form_id}.xml"
    if direct_match.exists():
        return direct_match

    matches = list((PROJECT_ROOT / "data" / "xml").rglob(f"{form_id}.xml"))
    if not matches:
        raise FileNotFoundError(f"Не найден XML для form id '{form_id}'.")
    return matches[0]


def to_int(value: str | None) -> int:
    if value is None:
        raise ValueError("В XML отсутствует обязательный числовой атрибут.")
    return int(value)


def union_boxes(boxes: Iterable[Box]) -> Box | None:
    box_list = list(boxes)
    if not box_list:
        return None

    return Box(
        left=min(box.left for box in box_list),
        top=min(box.top for box in box_list),
        right=max(box.right for box in box_list),
        bottom=max(box.bottom for box in box_list),
    )


def cmp_box(cmp_element: ET.Element) -> Box:
    x = to_int(cmp_element.get("x"))
    y = to_int(cmp_element.get("y"))
    width = to_int(cmp_element.get("width"))
    height = to_int(cmp_element.get("height"))
    return Box(left=x, top=y, right=x + width, bottom=y + height)


def contour_box(line_element: ET.Element) -> Box | None:
    points: list[tuple[int, int]] = []
    for contour_name in ("upper-contour", "lower-contour"):
        contour = line_element.find(contour_name)
        if contour is None:
            continue
        for point in contour.findall("point"):
            points.append((to_int(point.get("x")), to_int(point.get("y"))))

    if not points:
        return None

    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return Box(left=min(xs), top=min(ys), right=max(xs), bottom=max(ys))


def word_cmp_boxes(word_element: ET.Element) -> list[Box]:
    return [cmp_box(cmp_element) for cmp_element in word_element.findall("cmp")]


def word_box(word_element: ET.Element) -> Box | None:
    return union_boxes(word_cmp_boxes(word_element))


def line_box(line_element: ET.Element) -> Box | None:
    word_boxes = [box for word in line_element.findall("word") if (box := word_box(word))]
    contour = contour_box(line_element)
    if contour is not None:
        word_boxes.append(contour)
    return union_boxes(word_boxes)


def get_form_box(root: ET.Element) -> Box:
    width = to_int(root.get("width"))
    height = to_int(root.get("height"))
    return Box(left=0, top=0, right=width, bottom=height)


def find_line(root: ET.Element, line_id: str) -> ET.Element:
    line = root.find(f".//line[@id='{line_id}']")
    if line is None:
        raise LookupError(f"В XML не найдена строка '{line_id}'.")
    return line


def find_word(root: ET.Element, word_id: str) -> ET.Element:
    word = root.find(f".//word[@id='{word_id}']")
    if word is None:
        raise LookupError(f"В XML не найдено слово '{word_id}'.")
    return word


def build_annotations(root: ET.Element, target: ImageTarget) -> tuple[Box, list[Annotation], Annotation]:
    annotations: list[Annotation] = []

    if target.kind == "form":
        region_box = get_form_box(root)

        if DRAW_LINE_BOXES:
            for line in root.findall("./handwritten-part/line"):
                box = line_box(line)
                if box is None:
                    continue
                label = line.get("id") if DRAW_LABELS else None
                annotations.append(Annotation(box=box, color=COLORS["line"], label=label))

        if DRAW_WORD_BOXES:
            for word in root.findall(".//word"):
                box = word_box(word)
                if box is None:
                    continue
                label = word.get("text") if DRAW_LABELS else None
                annotations.append(Annotation(box=box, color=COLORS["word"], label=label))

        if DRAW_COMPONENT_BOXES:
            for word in root.findall(".//word"):
                for index, box in enumerate(word_cmp_boxes(word)):
                    label = f"cmp {index}" if DRAW_LABELS else None
                    annotations.append(Annotation(box=box, color=COLORS["cmp"], label=label))

        target_annotation = Annotation(
            box=region_box,
            color=COLORS["target"],
            label=target.entity_id if DRAW_LABELS else None,
        )
        return region_box, annotations, target_annotation

    if target.kind == "line":
        line = find_line(root, target.entity_id)
        region_box = line_box(line)
        if region_box is None:
            raise ValueError(f"Для строки '{target.entity_id}' не удалось собрать координаты.")

        if DRAW_WORD_BOXES:
            for word in line.findall("word"):
                box = word_box(word)
                if box is None:
                    continue
                label = word.get("text") if DRAW_LABELS else None
                annotations.append(Annotation(box=box, color=COLORS["word"], label=label))

        if DRAW_COMPONENT_BOXES:
            for word in line.findall("word"):
                for index, box in enumerate(word_cmp_boxes(word)):
                    label = f"cmp {index}" if DRAW_LABELS else None
                    annotations.append(Annotation(box=box, color=COLORS["cmp"], label=label))

        target_annotation = Annotation(
            box=region_box,
            color=COLORS["target"],
            label=target.entity_id if DRAW_LABELS else None,
        )
        return region_box, annotations, target_annotation

    if target.kind == "word":
        word = find_word(root, target.entity_id)
        region_box = word_box(word)
        if region_box is None:
            raise ValueError(f"Для слова '{target.entity_id}' не удалось собрать координаты.")

        if DRAW_COMPONENT_BOXES:
            for index, box in enumerate(word_cmp_boxes(word)):
                label = f"cmp {index}" if DRAW_LABELS else None
                annotations.append(Annotation(box=box, color=COLORS["cmp"], label=label))

        target_annotation = Annotation(
            box=region_box,
            color=COLORS["target"],
            label=word.get("text") if DRAW_LABELS else None,
        )
        return region_box, annotations, target_annotation

    word = find_word(root, target.entity_id)
    component_boxes = word_cmp_boxes(word)
    if target.component_index is None or target.component_index >= len(component_boxes):
        raise IndexError(
            f"У слова '{target.entity_id}' нет cmp с индексом {target.component_index}."
        )

    region_box = component_boxes[target.component_index]
    target_annotation = Annotation(
        box=region_box,
        color=COLORS["target"],
        label=f"cmp {target.component_index}" if DRAW_LABELS else None,
    )
    return region_box, annotations, target_annotation


def localize_box(box: Box, origin: Box, image_size: tuple[int, int]) -> Box:
    local_box = box.translated(-origin.left, -origin.top)
    scale_x = image_size[0] / origin.width
    scale_y = image_size[1] / origin.height
    return local_box.scaled(scale_x, scale_y)


def draw_annotation(draw: ImageDraw.ImageDraw, annotation: Annotation, font: ImageFont.ImageFont) -> None:
    draw.rectangle(
        [annotation.box.left, annotation.box.top, annotation.box.right, annotation.box.bottom],
        outline=annotation.color,
        width=LINE_WIDTH,
    )
    if not annotation.label:
        return

    text_x = max(annotation.box.left + 2, 0)
    text_y = max(annotation.box.top - 12, 0)
    draw.text((text_x, text_y), annotation.label, fill=annotation.color, font=font)


def get_output_path(image_path: Path) -> Path:
    annotated_name = f"{image_path.stem}_annotated{image_path.suffix}"

    if OUTPUT_PATH is None:
        return image_path.with_name(annotated_name)

    output_path = Path(OUTPUT_PATH)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if output_path.suffix:
        return output_path

    data_root = PROJECT_ROOT / "data"
    try:
        relative_parent = image_path.relative_to(data_root).parent
    except ValueError:
        relative_parent = Path()

    return output_path / relative_parent / annotated_name


def main() -> None:
    image_path = Path(IMAGE_PATH)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Не найдено изображение '{image_path}'.")

    target = parse_image_target(image_path)
    xml_path = find_xml_path(target.form_id)
    root = ET.parse(xml_path).getroot()

    image = Image.open(image_path).convert("RGB")
    region_box, annotations, target_annotation = build_annotations(root, target)

    if region_box.width <= 0 or region_box.height <= 0:
        raise ValueError(f"Некорректные размеры области разметки для '{target.entity_id}'.")

    localized_annotations = [
        Annotation(
            box=localize_box(annotation.box, region_box, image.size),
            color=annotation.color,
            label=annotation.label,
        )
        for annotation in annotations
    ]
    localized_target = Annotation(
        box=localize_box(target_annotation.box, region_box, image.size),
        color=target_annotation.color,
        label=target_annotation.label,
    )

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for annotation in localized_annotations:
        draw_annotation(draw, annotation, font)
    draw_annotation(draw, localized_target, font)

    output_path = get_output_path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"Image:  {image_path}")
    print(f"XML:    {xml_path}")
    print(f"Target: {target.kind} ({target.entity_id})")
    print(f"Saved:  {output_path}")


if __name__ == "__main__":
    main()
