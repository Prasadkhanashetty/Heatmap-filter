import numpy as np
from PIL import Image
from scipy import ndimage


def heat_view(image, z, mask, red_at, radius):
    """
    Overlays a smooth color heatmap (blue -> cyan -> green -> yellow -> red) 
    onto an image based on anomaly/intensity values.

    Args:
        image: Original PIL Image to draw on.
        z: 2D array of anomaly scores (same height & width as image).
        mask: 2D boolean array showing which pixels should be processed.
        red_at: The score threshold where the color turns pure red (0 to 12).
        radius: Controls how wide the soft glow/halo spreads around peaks (0 to 8).
    """
    # Make sure our threshold and radius are clean numbers
    red_at = float(red_at)
    radius = int(radius)

    # Sanity check: keep parameters within safe visual bounds
    if not np.isfinite(red_at) or not 0 <= red_at <= 12:
        raise ValueError("Red threshold must be between 0 and 12.")
    if not 0 <= radius <= 8:
        raise ValueError("Marker radius must be between 0 and 8.")

    # Convert inputs into standard NumPy arrays so we can do fast math on them
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    values = np.asarray(z, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    # Ensure the image, score map, and mask all line up pixel-for-pixel
    if values.shape != source.shape[:2] or mask.shape != values.shape:
        raise ValueError("Image, anomaly map and mask sizes must match.")

    # Ignore masked-out pixels or broken values (like NaN/infinity), and ignore negatives
    valid = mask & np.isfinite(values)
    values = np.where(valid, np.maximum(values, 0), 0)

    # Scale raw scores to a 0.0 - 1.0 intensity range using a fixed threshold
    # (Keeping this fixed ensures colors stay consistent across different images)
    if red_at > 0:
        strength = np.clip(values / red_at, 0, 1)
    else:
        strength = (values > 0).astype(np.float32)

    # Expand small defect dots into soft, visible halos so they are easy to spot
    if radius > 0:
        # First, dilate/expand the bright spots outward by 'radius' pixels
        spread = ndimage.maximum_filter(
            strength,
            size=2 * radius + 1,
            mode="constant",
            cval=0,
        )
        # Next, smooth the expanded edges with a Gaussian blur for a soft glow
        smooth = ndimage.gaussian_filter(
            spread,
            sigma=max(0.5, radius / 2),
            mode="constant",
            cval=0,
        )
        # Keep the sharpest original peaks while blending in the soft outer glow
        strength = np.maximum(strength, smooth)

    # Zero out anything outside our valid mask area
    strength = np.where(valid, np.clip(strength, 0, 1), 0)

    # Define our 5-color gradient stops from lowest intensity (0.0) to highest (1.0)
    stops = [0.0, 0.25, 0.50, 0.75, 1.0]
    colors = np.array([,   # Blue: very low deviation,   # Cyan: mild deviation,   # Green: moderate deviation,   # Yellow: high deviation,   # Red: critical peak deviation
    ], dtype=np.float32)

    # Map every pixel's 0-1 strength value to its corresponding RGB color
    heat_rgb = np.stack([
        np.interp(strength, stops, colors[:, channel])
        for channel in range(3)
    ], axis=-1)

    # Calculate transparency (alpha): keep normal areas clear, make defects bold
    # Using square root boosts faint spots slightly so subtle issues remain visible
    alpha = np.where(
        valid & (strength > 0),
        0.70 * np.sqrt(strength),
        0.0,
    )[..., None]

    # Blend the heatmap colors on top of the original photo
    overlay = source * (1.0 - alpha) + heat_rgb * alpha

    # Convert the final float math back into a standard 8-bit image
    return Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    )


def run_demo():
    """Loads a sample image, builds a test score map, and displays the heatmap."""
    # 1. Open the target image (swap 'input.jpg' with your own file path)
    input_path = "input.jpg"
    img = Image.open(input_path)
    width, height = img.size

    # 2. Build a dummy anomaly map (z) from brightness differences
    # (In production, replace this with your actual defect/model score array)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    z = np.abs(gray - np.mean(gray)) / (np.std(gray) + 1e-6) * 2.5

    # 3. Create an active mask covering the entire image
    mask = np.ones((height, width), dtype=bool)

    # 4. Generate the final heatmap overlay
    result = heat_view(
        image=img,
        z=z,
        mask=mask,
        red_at=6.0,   # Anything at or above 6.0 turns solid red
        radius=5      # Glow radius around hotspots
    )

    # 5. Save the output to disk and pop open the image viewer
    result.save("heatmap_output.png")
    result.show()
    print("Done! Saved result to heatmap_output.png")


if __name__ == "__main__":
    run_demo()
