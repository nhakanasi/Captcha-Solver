import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.ndimage import label

# ==== Parameters ====
MIN_WIDTH = 20  # Minimum width to allow splitting
PAD = 3        # Box padding

# ==== Step 1: Load and preprocess ====
image = cv2.imread(r'Self-taught\Captcha\captcha\captcha_0010.jpg', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
binary_bool = binary > 0
skeleton = skeletonize(binary_bool)
skeleton_img = (skeleton * 255).astype(np.uint8)

h, w = skeleton.shape

# ==== Step 2: Compute degree of each skeleton pixel ====
degree_map = np.zeros_like(skeleton, dtype=np.uint8)
neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

for y in range(h):
    for x in range(w):
        if skeleton[y, x]:
            degree = 0
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                    degree += 1
            degree_map[y, x] = degree

# Branch points where degree >= 3
branch_points = ((degree_map >= 3) & (degree_map < 4)).astype(np.uint8)


# ==== Step 3: Connected components for non-touching digits ====
cc_labeled, cc_count = label(binary_bool)
cc_boxes = []
for i in range(1, cc_count + 1):
    ys, xs = np.where(cc_labeled == i)
    if len(xs) > 0:
        cc_boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))

# ==== Step 4: For each CC, check for branch clusters and split if needed ====
output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
output_skeleton = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)

all_boxes = []

for (x_min_cc, y_min_cc, x_max_cc, y_max_cc) in cc_boxes:
    # Extract CC region
    cc_segment = skeleton[y_min_cc:y_max_cc+1, x_min_cc:x_max_cc+1]
    cc_branch = branch_points[y_min_cc:y_max_cc+1, x_min_cc:x_max_cc+1]

    # Find branch clusters inside CC
    labeled_branch, num_branch = label(cc_branch)
    cut_positions = []
    for j in range(1, num_branch + 1):
        ys, xs = np.where(labeled_branch == j)
        if len(xs) > 0:
            cut_x = int(np.mean(xs)) + x_min_cc
            cut_positions.append(cut_x)

    cut_positions = sorted(cut_positions)

    # Always include full CC width as last segment
    cuts = [x_min_cc] + cut_positions + [x_max_cc + 1]

    # Build boxes from cuts
    prev_cut = cuts[0]
    for cut in cuts[1:]:
        if cut - prev_cut >= MIN_WIDTH:  # Only cut if wide enough
            seg_x_min = prev_cut
            seg_x_max = cut
            seg_y_min = y_min_cc
            seg_y_max = y_max_cc

            # Expand box
            seg_x_min = max(0, seg_x_min - PAD)
            seg_x_max = min(w, seg_x_max + PAD)
            seg_y_min = max(0, seg_y_min - PAD)
            seg_y_max = min(h, seg_y_max + PAD)

            all_boxes.append((seg_x_min, seg_y_min, seg_x_max, seg_y_max))

            # Draw on both images
            cv2.rectangle(output_img, (seg_x_min, seg_y_min), (seg_x_max, seg_y_max), (0, 0, 255), 2)
            cv2.rectangle(output_skeleton, (seg_x_min, seg_y_min), (seg_x_max, seg_y_max), (0, 255, 0), 2)

        prev_cut = cut

# Highlight branch points in blue on skeleton
branch_y, branch_x = np.where(branch_points)
for (bx, by) in zip(branch_x, branch_y):
    cv2.circle(output_skeleton, (bx, by), 1, (255, 0, 0), -1)

# ==== Step 5: Visualize ====
plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
plt.title("Original with Boxes")
plt.imshow(output_img)
plt.axis("off")   # turn off x/y axis

plt.subplot(1, 2, 2)
plt.title("Skeleton with Boxes & Branch Points")
plt.imshow(output_skeleton)
plt.axis("off")   # turn off x/y axis

plt.show()
