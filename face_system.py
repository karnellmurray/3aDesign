import os
from pathlib import Path
from facenet_pytorch import MTCNN
import torch
from PIL import Image, ImageDraw
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt


# Set device and initialize MTCNN
device = torch.device('cpu')
mtcnn = MTCNN(keep_all=True, device='cpu')

# Extracting embeddings
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')

# Preprocess for FaceNet
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# --- S T E P   O N E   :   A L I G N   F A C E S ---
def align_faces(input_root="face_detection", output_root="aligned_faces"):
    # Takes the input root and output root of the folder where the face images you're using are
    # Outputs the cropped and aligned faces to the output folder you define (aligned_faces)
    input_root = Path(input_root)
    output_root = Path(output_root)
    # Gets the images from the folder you've requested (face_detection > Team name > Team player > Player)
    for team_dir in input_root.iterdir():
        if team_dir.is_dir():
            for player_dir in team_dir.iterdir():
                if player_dir.is_dir():
                    team_name = team_dir.name
                    player_name = player_dir.name
                    # Tells you which player is being processed, the team name and the player name
                    print(f"Processing {team_name}/{player_name}...")
                    # Outputs the new directory path
                    output_player_dir = output_root / team_name / player_name
                    # If the directory doesnt exist it will create it
                    output_player_dir.mkdir(parents=True, exist_ok=True)
                    # Takes JPG, JPEG and PNG
                    image_paths = list(player_dir.glob("*.jpg")) + \
                                  list(player_dir.glob("*.jpeg")) + \
                                  list(player_dir.glob("*.png"))
                    # Converts image to RGB to detect face 
                    for image_path in image_paths:
                        img = Image.open(image_path).convert("RGB")
                        aligned_faces = mtcnn(img)

                        if aligned_faces is None:
                            print(f"No face detected in {image_path.name}")
                            continue

                        if isinstance(aligned_faces, torch.Tensor) and aligned_faces.ndim == 4:
                            for i in range(aligned_faces.size(0)):
                                face_tensor = aligned_faces[i]
                                unnorm = face_tensor * 0.5 + 0.5
                                face_img = transforms.ToPILImage()(unnorm)

                                if aligned_faces.size(0) > 1:
                                    save_name = f"{image_path.stem}_face{i+1}{image_path.suffix}"
                                else:
                                    save_name = image_path.name

                                save_path = output_player_dir / save_name
                                face_img.save(save_path)
                                print(f"Saved aligned face: {save_path}")


# --- S T E P   T W O   :   C R E A T E   E M B E D D I N G S ---
# Creates player embeddings (details of the player faces using the aligned faces from step 1)
# Check the aligned faces before calling this step as sometimes the software detects blurs or body parts as faces
def create_embeddings(input_root="aligned_faces"):
    # Input where the aligned faces are, so you're using the correct images for embedding creation
    input_root = Path(input_root)

    for team_dir in input_root.iterdir():
        if team_dir.is_dir():
            for player_dir in team_dir.iterdir():
                if player_dir.is_dir():
                    print(f"\nEmbedding faces for {team_dir.name}/{player_dir.name}")

                    for img_path in player_dir.glob("*"):
                        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            img = Image.open(img_path).convert("RGB")
                            tensor = preprocess(img).unsqueeze(0).to(device)

                            with torch.no_grad():
                                emb = resnet(tensor).squeeze().cpu().numpy()

                            emb_path = img_path.with_suffix(".npy")
                            np.save(emb_path, emb)
                            print(f"Saved embedding: {emb_path}")


# --- S T E P   T H R E E   :   C R E A T E   P L A Y E R   T E M P L A T E S ---
# Each player will have multiple embeddings used to create a player template for higher matching
def create_templates(embedding_root="aligned_faces", template_output_root="player_templates"):
    embedding_root = Path(embedding_root)
    template_output_root = Path(template_output_root)
    template_output_root.mkdir(exist_ok=True)

    for team_dir in embedding_root.iterdir():
        if team_dir.is_dir():
            for player_dir in team_dir.iterdir():
                if player_dir.is_dir():
                    player_name = f"{team_dir.name}_{player_dir.name}"
                    player_template_dir = template_output_root / player_name
                    player_template_dir.mkdir(parents=True, exist_ok=True)

                    embedding_files = list(player_dir.glob("*.npy"))
                    if not embedding_files:
                        print(f"No embeddings found for {player_name}")
                        continue

                    for emb_file in embedding_files:
                        emb = np.load(emb_file)
                        if emb.shape != (512,):
                            continue

                        save_path = player_template_dir / emb_file.name
                        np.save(save_path, emb)
                        print(f"Saved template: {save_path}")

# --- S T E P   F O U R   :   F I N D   M A T C H E S ---
# An anchor is the first instance a player has been seen, so its not just being matched against photos but also the first accurate frame in the video itself
anchors = {}
# Player matching against threshold, hight threshold = higher confidence
def match_face_to_players_with_anchor(face_embedding, templates, threshold=0.6, anchor_threshold=0.8):
    best_score = -1
    best_player = "Unknown"

    for player, views in templates.items():
        sims = [
            cosine_similarity(
                torch.tensor(face_embedding),
                torch.tensor(template),
                dim=0
            ).item()
            for _, template in views
        ]
        avg_sim = sum(sims) / len(sims)
        if avg_sim > best_score:
            best_score = avg_sim
            best_player = player

    if best_score < threshold:
        return "Unknown", best_score, False

    if best_player in anchors:
        anchor_sim = cosine_similarity(
            torch.tensor(face_embedding),
            torch.tensor(anchors[best_player]),
            dim=0
        ).item()
        return (best_player, best_score, anchor_sim >= anchor_threshold)
    else:
        anchors[best_player] = face_embedding
        return (best_player, best_score, True)
# Run the video you want to find faces to match against photos
def run_video(input_video="video/4k-output.mp4", output_video="output.mp4", frame_skip=3):
    # Load templates
    player_templates = {}
    template_dir = Path("player_templates")
    for player_dir in template_dir.iterdir():
        if player_dir.is_dir():
            templates = []
            for emb_file in player_dir.glob("*.npy"):
                emb = np.load(emb_file)
                if emb.shape == (512,):
                    templates.append((emb_file.stem, emb))
            if templates:
                player_templates[player_dir.name] = templates

    # Video setup
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        boxes, _ = mtcnn.detect(pil_img)

        if boxes is not None:
            for box in boxes:
                box = [int(b) for b in box]
                face_crop = pil_img.crop(box)
                face_tensor = preprocess(face_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = resnet(face_tensor).squeeze().cpu().numpy()

                best_player, best_score, passed_anchor = match_face_to_players_with_anchor(
                    embedding, player_templates, threshold=0.6, anchor_threshold=0.8
                )

                if passed_anchor and best_player != "Unknown":
                    color = (0, 255, 0)
                    label = f"{best_player} ({best_score:.2f})"

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, label, (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()