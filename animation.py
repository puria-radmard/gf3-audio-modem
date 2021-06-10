from glob import glob
import cv2, os

folder = "videos_reestimate"

frames = glob(f"{folder}/updated_channel*.png")
frames.sort(key = lambda x: int(x.split("/updated_channel")[-1].split(".png")[0]))

frame = cv2.imread(frames[0])
height, width, layers = frame.shape
video = cv2.VideoWriter(
    f"{folder}/pilot_update.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    3,
    (width, height),
)

for image in frames:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()

# for file_name in glob.glob(f"{folder}/updated_channel*.png"):    
#     os.remove(file_name)