#!/usr/bin/env python3
"""Local browser GUI for human review of SeatVision box annotations.

The vocabulary is intentionally restricted to chair, table, and object. This
is a fast box-level bootstrap for a custom detector, not seat-surface mask
annotation or a final occupancy policy.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES_ROOT = SCRIPT_ROOT / "datasets/raw_videos/phone/prepared/object_claim_2fps"
DEFAULT_PRELABELS_ROOT = SCRIPT_ROOT / "datasets/raw_videos/phone/prelabels/object_claim_2fps"
DEFAULT_OUTPUT = SCRIPT_ROOT / "datasets/raw_videos/phone/human_annotations/object_claim_2fps.json"
LABELS = ("chair", "table", "object")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review SeatVision chair/table/object box annotations locally.")
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--prelabels-root", type=Path, default=DEFAULT_PRELABELS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Local editable annotation JSON")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host; default keeps labels local to the Jetson")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true", help="Open the GUI in the Jetson's default browser")
    args = parser.parse_args()
    args.frames_root = args.frames_root.resolve()
    args.prelabels_root = args.prelabels_root.resolve()
    args.output = args.output.resolve()
    if not args.frames_root.is_dir():
        parser.error(f"Frames root not found: {args.frames_root}")
    if not args.prelabels_root.is_dir():
        parser.error(f"Prelabels root not found: {args.prelabels_root}")
    if not (1 <= args.port <= 65535):
        parser.error("--port must be between 1 and 65535")
    return args


def json_write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def map_model_label(detection: dict[str, Any]) -> str:
    if detection.get("semantic_role") == "seat":
        return "chair"
    if detection.get("label") == "dining table":
        return "table"
    return "object"


def load_prelabels(prelabels_root: Path, frames_root: Path) -> dict[str, list[dict[str, Any]]]:
    mapped: dict[str, list[dict[str, Any]]] = {}
    for predictions_path in sorted(prelabels_root.glob("*/predictions.jsonl")):
        session_id = predictions_path.parent.name
        for line_number, line in enumerate(predictions_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                prediction = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {predictions_path}:{line_number}") from exc
            source = prediction.get("source")
            if not isinstance(source, str):
                continue
            key = f"{session_id}/frames/{source}"
            frame_path = frames_root / key
            if not frame_path.is_file():
                continue
            annotations = mapped.setdefault(key, [])
            for detection_index, detection in enumerate(prediction.get("detections", [])):
                box = detection.get("bbox_xywh")
                if not isinstance(box, list) or len(box) != 4:
                    continue
                annotations.append(
                    {
                        "id": f"prelabel-{session_id}-{source}-{detection_index}",
                        "label": map_model_label(detection),
                        "bbox_xywh": [float(value) for value in box],
                    }
                )
    return mapped


def build_dataset(frames_root: Path, prelabels_root: Path) -> dict[str, Any]:
    proposals = load_prelabels(prelabels_root, frames_root)
    images: list[dict[str, Any]] = []
    for image_path in sorted(path for path in frames_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            print(f"Skipping undecodable image: {image_path}")
            continue
        key = safe_relative(image_path, frames_root)
        images.append(
            {
                "id": key,
                "file": key,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "reviewed": False,
                "annotations": proposals.get(key, []),
            }
        )
    if not images:
        raise RuntimeError(f"No images found under: {frames_root}")
    return {
        "schema_version": "seatvision-box-annotations/v1",
        "annotation_note": (
            "Reviewed boxes only. Labels are chair, table, object. "
            "These boxes do not establish seat-surface segmentation or occupancy state."
        ),
        "frames_root": str(frames_root),
        "categories": list(LABELS),
        "images": images,
    }


def simplify_annotation_metadata(dataset: dict[str, Any]) -> bool:
    """Discard model/provenance subtype fields from an existing local review file.

    The reviewer sees and edits only the three agreed classes. Keeping detector
    names such as cup/handbag or an edit-source field would make the annotation
    task look like a finer-grained classification task when it is not.
    """
    changed = False
    for image in dataset.get("images", []):
        simplified: list[dict[str, Any]] = []
        for annotation in image.get("annotations", []):
            clean = {
                "id": annotation.get("id"),
                "label": annotation.get("label"),
                "bbox_xywh": annotation.get("bbox_xywh"),
            }
            if annotation != clean:
                changed = True
            simplified.append(clean)
        image["annotations"] = simplified
    return changed


def validate_dataset(dataset: dict[str, Any]) -> None:
    if dataset.get("schema_version") != "seatvision-box-annotations/v1":
        raise ValueError("Unsupported annotation schema")
    if dataset.get("categories") != list(LABELS):
        raise ValueError("Annotation categories must be exactly chair, table, object")
    images = dataset.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Annotation dataset contains no images")
    for image in images:
        if not isinstance(image.get("file"), str) or image["file"].startswith("/") or ".." in Path(image["file"]).parts:
            raise ValueError("Unsafe image path in annotation dataset")
        width, height = image.get("width"), image.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            raise ValueError("Invalid image dimensions")
        if not isinstance(image.get("reviewed"), bool):
            raise ValueError("Image reviewed state must be boolean")
        annotations = image.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError("Image annotations must be a list")
        seen_ids: set[str] = set()
        for annotation in annotations:
            if annotation.get("label") not in LABELS:
                raise ValueError("Annotation uses an unsupported label")
            annotation_id = annotation.get("id")
            if not isinstance(annotation_id, str) or annotation_id in seen_ids:
                raise ValueError("Annotation IDs must be unique strings per image")
            seen_ids.add(annotation_id)
            box = annotation.get("bbox_xywh")
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) for value in box):
                raise ValueError("Invalid annotation rectangle")
            x, y, box_width, box_height = box
            if box_width <= 0 or box_height <= 0 or x < 0 or y < 0 or x + box_width > width or y + box_height > height:
                raise ValueError("Annotation rectangle lies outside image")


def coco_export(dataset: dict[str, Any]) -> dict[str, Any]:
    categories = [{"id": index + 1, "name": label, "supercategory": "seat_scene"} for index, label in enumerate(LABELS)]
    category_ids = {category["name"]: category["id"] for category in categories}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, image in enumerate(dataset["images"], start=1):
        if not image["reviewed"]:
            continue
        images.append({"id": image_id, "file_name": image["file"], "width": image["width"], "height": image["height"]})
        for annotation in image["annotations"]:
            x, y, width, height = annotation["bbox_xywh"]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[annotation["label"]],
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {"description": "SeatVision human-reviewed chair/table/object boxes"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


GUI_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SeatVision training-data review</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;background:#111820;color:#eef3f8}
header{display:flex;gap:.7rem;align-items:center;padding:.7rem 1rem;background:#1c2733;position:sticky;top:0;z-index:2}
button{border:0;border-radius:.35rem;padding:.5rem .7rem;background:#314357;color:#fff;font-weight:600;cursor:pointer}button:hover{background:#45617f}button.active{outline:3px solid #fff}button.chair{background:#287be0}button.table{background:#c9a300}button.object{background:#b64ac9}button.danger{background:#a33d47}.status{margin-left:auto;color:#b9ccdd}.layout{display:grid;grid-template-columns:minmax(0,1fr) 340px;min-height:calc(100vh - 58px)}
main{padding:1rem;overflow:auto}.canvas-wrap{display:inline-block;max-width:100%;background:#000;border:1px solid #3c4c5f}canvas{display:block;max-width:100%;height:auto;cursor:crosshair}aside{padding:1rem;background:#17212c;border-left:1px solid #344454;overflow:auto}.help{font-size:.85rem;color:#b8c7d5;line-height:1.45}.badge{display:inline-block;padding:.12rem .35rem;border-radius:.25rem;background:#28394a;font-size:.78rem}.annotation{width:100%;text-align:left;margin:.3rem 0;padding:.5rem;background:#263545}.annotation.selected{outline:2px solid white}.annotation small{display:block;color:#c0cfdd;margin-top:.2rem}.reviewed{color:#71e69c}.unreviewed{color:#ffca75}.empty{color:#9fb0bf;font-style:italic}
</style></head><body>
<header><button id="previous">← Previous</button><button id="next">Next →</button><button id="review">Mark reviewed</button><button id="save">Save changes</button><button id="export">Export reviewed COCO</button><span class="status" id="status">Loading…</span></header>
<div class="layout"><main><div class="canvas-wrap"><canvas id="canvas"></canvas></div></main><aside>
<h2 id="title">Training-data review</h2><p class="help">Click a box outline to select it. The inside of a box remains drawable, so boxes may partially or fully overlap. Click a label to change the selected box. Drag to draw a new box. Delete removes the selected box. Keyboard: <kbd>C</kbd> chair, <kbd>T</kbd> table, <kbd>O</kbd> object, <kbd>Delete</kbd>, <kbd>←</kbd>/<kbd>→</kbd>, <kbd>Ctrl+S</kbd>.</p>
<div><button class="chair" data-label="chair">Chair (C)</button><button class="table" data-label="table">Table (T)</button><button class="object" data-label="object">Object (O)</button><button class="danger" id="delete">Delete</button></div>
<p class="help">Only mark a frame reviewed after correcting or intentionally accepting every useful box. Prelabels are suggestions; they are not ground truth.</p><h3>Boxes</h3><div id="annotations"></div></aside></div>
<script>
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
let dataset, current=0, selected=null, drawing=null, defaultLabel='object', image=new Image(), dirty=false;
const colors={chair:'#2385ff',table:'#f2c300',object:'#df50ef'};
function entry(){return dataset.images[current]}
function annotation(){return entry().annotations.find(a=>a.id===selected)}
function imageUrl(){return '/frames/'+encodeURIComponent(entry().file).replaceAll('%2F','/')}
function setStatus(text){document.getElementById('status').textContent=text}
function markDirty(){dirty=true; setStatus('Unsaved changes')}
// An interior click must remain drawable over a large parent box (for example,
// an object on a table). Only the visible rectangle outline is selectable.
// The tolerance is measured in native image pixels after point() conversion.
const OUTLINE_HIT_TOLERANCE=12;
function bboxAt(x,y){for(const box of [...entry().annotations].reverse()){const [bx,by,bw,bh]=box.bbox_xywh;const withinX=x>=bx-OUTLINE_HIT_TOLERANCE&&x<=bx+bw+OUTLINE_HIT_TOLERANCE;const withinY=y>=by-OUTLINE_HIT_TOLERANCE&&y<=by+bh+OUTLINE_HIT_TOLERANCE;if(!withinX||!withinY)continue;const onVertical=Math.abs(x-bx)<=OUTLINE_HIT_TOLERANCE||Math.abs(x-(bx+bw))<=OUTLINE_HIT_TOLERANCE;const onHorizontal=Math.abs(y-by)<=OUTLINE_HIT_TOLERANCE||Math.abs(y-(by+bh))<=OUTLINE_HIT_TOLERANCE;if(onVertical||onHorizontal)return box}return null}
function point(event){const r=canvas.getBoundingClientRect();return {x:(event.clientX-r.left)*canvas.width/r.width,y:(event.clientY-r.top)*canvas.height/r.height}}
function draw(){if(!dataset||!image.complete)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);for(const box of entry().annotations){const [x,y,w,h]=box.bbox_xywh;ctx.save();ctx.strokeStyle=colors[box.label];ctx.lineWidth=selected===box.id?6:3;ctx.strokeRect(x,y,w,h);ctx.font='bold 22px sans-serif';const text=box.label;const tw=ctx.measureText(text).width+12;ctx.fillStyle=colors[box.label];ctx.fillRect(x,Math.max(0,y-28),tw,28);ctx.fillStyle='#111';ctx.fillText(text,x+6,Math.max(22,y-7));ctx.restore()}if(drawing){const{x,y,w,h}=drawing;ctx.save();ctx.strokeStyle=colors[defaultLabel];ctx.lineWidth=3;ctx.setLineDash([8,5]);ctx.strokeRect(x,y,w,h);ctx.restore()}renderList()}
function renderList(){const holder=document.getElementById('annotations');holder.replaceChildren();if(!dataset)return;const boxes=entry().annotations;if(!boxes.length){holder.innerHTML='<p class="empty">No boxes. Drag to add an annotation.</p>';return}for(const box of boxes){const button=document.createElement('button');button.className='annotation '+(selected===box.id?'selected':'');button.innerHTML='<span class="badge">'+box.label+'</span><small>'+box.bbox_xywh.map(v=>Math.round(v)).join(', ')+'</small>';button.onclick=()=>{selected=box.id;draw()};holder.append(button)}}
function loadImage(){selected=null;const e=entry();document.getElementById('title').textContent=(current+1)+' / '+dataset.images.length+' · '+e.file;document.getElementById('review').textContent=e.reviewed?'Reviewed ✓':'Mark reviewed';document.getElementById('review').className=e.reviewed?'reviewed':'unreviewed';setStatus(dirty?'Unsaved changes':(e.reviewed?'Reviewed':'Needs review'));image=new Image();image.onload=()=>{canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;draw()};image.src=imageUrl()}
function applyLabel(label){defaultLabel=label;document.querySelectorAll('[data-label]').forEach(b=>b.classList.toggle('active',b.dataset.label===label));const box=annotation();if(box){box.label=label;markDirty();draw()}}
function deleteSelected(){if(!selected)return;entry().annotations=entry().annotations.filter(a=>a.id!==selected);selected=null;markDirty();draw()}
async function save(){const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(dataset)});const body=await response.json();if(!response.ok)throw Error(body.error||'save failed');dirty=false;setStatus('Saved '+body.path)}
async function exportCoco(){const response=await fetch('/api/export-coco',{method:'POST'});const body=await response.json();if(!response.ok)throw Error(body.error||'export failed');setStatus('Exported '+body.path+' · '+body.images+' reviewed images')}
canvas.addEventListener('mousedown',event=>{const p=point(event),hit=bboxAt(p.x,p.y);if(hit){selected=hit.id;drawing=null;draw();return}selected=null;drawing={x:p.x,y:p.y,w:0,h:0};draw()});
canvas.addEventListener('mousemove',event=>{if(!drawing)return;const p=point(event);drawing.w=p.x-drawing.x;drawing.h=p.y-drawing.y;draw()});
canvas.addEventListener('mouseup',event=>{if(!drawing)return;const p=point(event);let{x,y,w,h}=drawing;w=p.x-x;h=p.y-y;if(w<0){x+=w;w=-w}if(h<0){y+=h;h=-h}drawing=null;if(w>=8&&h>=8){const box={id:'box-'+crypto.randomUUID(),label:defaultLabel,bbox_xywh:[x,y,w,h]};entry().annotations.push(box);selected=box.id;markDirty()}draw()});
document.getElementById('previous').onclick=()=>{if(current>0){current--;loadImage()}};document.getElementById('next').onclick=()=>{if(current<dataset.images.length-1){current++;loadImage()}};document.getElementById('review').onclick=()=>{entry().reviewed=!entry().reviewed;markDirty();loadImage()};document.getElementById('save').onclick=()=>save().catch(e=>setStatus('Save error: '+e.message));document.getElementById('export').onclick=()=>exportCoco().catch(e=>setStatus('Export error: '+e.message));document.getElementById('delete').onclick=deleteSelected;document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>applyLabel(b.dataset.label));
window.addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();save().catch(e=>setStatus('Save error: '+e.message));return}if(event.target.tagName==='INPUT')return;const key=event.key.toLowerCase();if(key==='c')applyLabel('chair');if(key==='t')applyLabel('table');if(key==='o')applyLabel('object');if(event.key==='Delete'||event.key==='Backspace')deleteSelected();if(event.key==='ArrowLeft'&&current>0){current--;loadImage()}if(event.key==='ArrowRight'&&current<dataset.images.length-1){current++;loadImage()}});
fetch('/api/dataset').then(r=>r.json()).then(data=>{dataset=data;applyLabel(defaultLabel);loadImage()}).catch(e=>setStatus('Load error: '+e.message));
</script></body></html>"""


class AnnotationApplication:
    def __init__(self, frames_root: Path, output: Path, dataset: dict[str, Any]) -> None:
        self.frames_root = frames_root
        self.output = output
        self.dataset = dataset
        self.lock = threading.Lock()

    def save(self, value: dict[str, Any]) -> None:
        validate_dataset(value)
        with self.lock:
            self.dataset = value
            json_write_atomic(self.output, self.dataset)

    def export_coco(self) -> Path:
        with self.lock:
            export_path = self.output.with_suffix(".coco.json")
            json_write_atomic(export_path, coco_export(self.dataset))
            return export_path


def make_handler(application: AnnotationApplication):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            print(f"annotation-gui: {fmt % args}")

        def send_json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            payload = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                payload = GUI_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/api/dataset":
                with application.lock:
                    self.send_json(HTTPStatus.OK, application.dataset)
                return
            if path.startswith("/frames/"):
                requested = Path(unquote(path.removeprefix("/frames/")))
                try:
                    file_path = (application.frames_root / requested).resolve()
                    file_path.relative_to(application.frames_root)
                except ValueError:
                    self.send_error(HTTPStatus.FORBIDDEN, "Invalid frame path")
                    return
                if not file_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Frame not found")
                    return
                content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
                payload = file_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/export-coco":
                try:
                    exported = application.export_coco()
                    with application.lock:
                        reviewed = sum(1 for image in application.dataset["images"] if image["reviewed"])
                    self.send_json(HTTPStatus.OK, {"path": str(exported), "images": reviewed})
                except (OSError, ValueError) as exc:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 50_000_000:
                    raise ValueError("Invalid save payload size")
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                application.save(value)
                self.send_json(HTTPStatus.OK, {"path": str(application.output)})
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return Handler


def main() -> int:
    args = parse_arguments()
    if args.output.is_file():
        dataset = json.loads(args.output.read_text(encoding="utf-8"))
        simplified = simplify_annotation_metadata(dataset)
        validate_dataset(dataset)
        if simplified:
            json_write_atomic(args.output, dataset)
        print(f"Resuming annotations: {args.output}")
    else:
        dataset = build_dataset(args.frames_root, args.prelabels_root)
        json_write_atomic(args.output, dataset)
        print(f"Created annotations from proposals: {args.output}")
    application = AnnotationApplication(args.frames_root, args.output, dataset)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(application))
    url = f"http://{args.host}:{args.port}/"
    print(f"SeatVision annotation GUI: {url}")
    print("Use Ctrl+C in this terminal to stop the local server.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnnotation GUI stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
