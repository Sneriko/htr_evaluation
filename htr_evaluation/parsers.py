from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


XMLFormat = str


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def detect_xml_format(xml_path: str | Path) -> XMLFormat:
    """Detect whether an XML file is ALTO, PAGE XML, or unknown."""
    path = Path(xml_path)
    for _, root in ET.iterparse(path, events=("start",)):
        tag = _local_name(root.tag).lower()
        if tag == "alto":
            return "alto"
        if tag == "pcgts":
            return "page"
        break
    return "unknown"


def _extract_from_alto(root: ET.Element) -> str:
    parts: list[str] = []
    for elem in root.iter():
        if _local_name(elem.tag) == "String":
            content = elem.attrib.get("CONTENT")
            if content:
                parts.append(content)
    return " ".join(parts)


def _extract_from_pagexml(root: ET.Element) -> str:
    parts: list[str] = []
    for elem in root.iter():
        if _local_name(elem.tag) == "Unicode":
            text = "".join(elem.itertext()).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def extract_text(xml_path: str | Path) -> str:
    """
    Extract readable text from a PAGE XML or ALTO XML document.
    """
    path = Path(xml_path)
    tree = ET.parse(path)
    root = tree.getroot()

    tag = _local_name(root.tag).lower()
    if tag == "alto":
        return _extract_from_alto(root)
    if tag == "pcgts":
        return _extract_from_pagexml(root)

    # Best-effort fallback for unknown but compatible structures.
    page_text = _extract_from_pagexml(root)
    if page_text:
        return page_text

    alto_text = _extract_from_alto(root)
    if alto_text:
        return alto_text

    return " ".join(t.strip() for t in root.itertext() if t.strip())
