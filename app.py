from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# Load trained model
model = YOLO("C:/Users/asifm/road_damage_detection/runs/detect/train/weights/best.pt")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -----------------------------
# ENGINEERING CONSTANTS
# -----------------------------
POTHOLE_DEPTH = 0.04
CRACK_DEPTH = 0.015
PATCH_DEPTH = 0.03

ASPHALT_DENSITY = 2400
COST_PER_M3 = 40000

SHAPE_FACTOR = 0.4


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        files = request.files.getlist("images")

        results_summary = []
        uploaded_images = []
        result_images = []

        batch_total_volume = 0
        batch_overall_severity = "Low"

        for file in files:
            if file.filename == "":
                continue

            # Save uploaded image
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            uploaded_images.append(file.filename)

            image = cv2.imread(filepath)
            img_height, img_width, _ = image.shape

            pixels_per_meter = (img_width * 0.65) / 3.5

            # Run detection
            results = model(filepath, conf=0.25)
            boxes = results[0].boxes

            total_volume = 0
            overall_severity = "Low"

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])

                    x1, y1, x2, y2 = box.xyxy[0]

                    pixel_width = float(x2 - x1)
                    pixel_height = float(y2 - y1)

                    width_m = pixel_width / pixels_per_meter
                    height_m = pixel_height / pixels_per_meter
                    area = width_m * height_m

                    # -----------------------------
                    # DAMAGE LOGIC
                    # -----------------------------
                    if cls == 0:  # Pothole
                        volume = area * POTHOLE_DEPTH * SHAPE_FACTOR

                        if area < 0.05:
                            severity = "Low"
                        elif area < 0.20:
                            severity = "Medium"
                        else:
                            severity = "Severe"

                    elif cls == 1:  # Crack
                        crack_length = max(width_m, height_m)
                        volume = crack_length * 0.05 * CRACK_DEPTH

                        if crack_length < 0.5:
                            severity = "Low"
                        elif crack_length < 2:
                            severity = "Medium"
                        else:
                            severity = "Severe"

                    elif cls == 2:  # Patch
                        volume = area * PATCH_DEPTH * SHAPE_FACTOR

                        if area < 0.10:
                            severity = "Low"
                        elif area < 0.40:
                            severity = "Medium"
                        else:
                            severity = "Severe"
                    else:
                        continue

                    # Severity priority logic
                    if severity == "Severe":
                        overall_severity = "Severe"
                    elif severity == "Medium" and overall_severity != "Severe":
                        overall_severity = "Medium"

                    total_volume += volume

            # Calculations
            total_mass = total_volume * ASPHALT_DENSITY
            total_cost = total_volume * COST_PER_M3

            batch_total_volume += total_volume

            if overall_severity == "Severe":
                batch_overall_severity = "Severe"
            elif overall_severity == "Medium" and batch_overall_severity != "Severe":
                batch_overall_severity = "Medium"

            # Save result image
            result_path = os.path.join(RESULT_FOLDER, file.filename)
            annotated = results[0].plot()
            cv2.imwrite(result_path, annotated)

            result_images.append(file.filename)

            # Store summary
            results_summary.append({
                "filename": file.filename,
                "mass": round(total_mass, 2),
                "cost": round(total_cost, 2),
                "severity": overall_severity
            })

        # Batch totals
        batch_total_mass = batch_total_volume * ASPHALT_DENSITY
        batch_total_cost = batch_total_volume * COST_PER_M3

        print("Uploaded Images:", uploaded_images)
        print("Result Images:", result_images)
        print("Summary:", results_summary)
        return render_template(
            "batch_result.html",
            results_summary=results_summary,
            uploaded_images=uploaded_images,
            result_images=result_images,
            batch_total_mass=round(batch_total_mass, 2),
            batch_total_cost=round(batch_total_cost, 2),
            batch_overall_severity=batch_overall_severity
        )

    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(debug=True)