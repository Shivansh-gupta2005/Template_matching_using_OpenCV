import numpy as np
import cv2
import argparse

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def run(sourceimg, templateimg, outputfilename):
    # Reading template image
    template = cv2.imread(templateimg, 0)
    cv2.imshow("Template Image", template)  # Show the template image

    # Applying Laplacian transformation to the template
    template = cv2.Laplacian(template, cv2.CV_64F)
    template = np.float32(template)
    (tH, tW) = template.shape[:2]

    # Reading source image
    image = cv2.imread(sourceimg)
    cv2.imshow("Source Image", image)  # Show the source image

    # Convert source image to grayscale and apply preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.Laplacian(blur, cv2.CV_64F)
    gray = np.float32(gray)
    found = None

    # Template matching loop for multiple scales
    for scale in np.linspace(0.5, 2, 30):
        resized = resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # Draw bounding box and display results
    if found is not None:
        (maxVal, maxLoc, r) = found
        if maxVal > 350000:
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.imshow("Output Image", image)  # Show the output image
            cv2.imwrite(f"{outputfilename}.png", image)
        else:
            print(sourceimg, "Cursor not detected")
    else:
        print(sourceimg, "Cursor not detected")

    # Wait for key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Template extraction")

    parser.add_argument(
        "-s",
        "--source",
        help="Source image",
        type=str,
        default="source.png",
        required=False,
    )

    parser.add_argument(
        "-t",
        "--template",
        help="Template image (i.e. that you want to search) Note: Please make sure that the template image is smaller than the source image.",
        type=str,
        default="template.png",
        required=False,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Name of output file",
        type=str,
        default="output",
        required=False,
    )

    args = parser.parse_args()
    run(sourceimg=args.source, templateimg=args.template, outputfilename=args.output)
