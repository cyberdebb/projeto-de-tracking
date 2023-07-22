#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

const char* params = 
"{ input | pessoas.mp4 | Path to a video or a sequence of images }"
"{ algo | MOG2 | Background subtraction method (KNN, MOG2) }";

int main(int argc, char* argv[])
{
    namedWindow ("final", WINDOW_NORMAL);
    namedWindow ("background subtraction", WINDOW_NORMAL);

    CommandLineParser parser(argc, argv, params);

    // create Background Subtractor object
    Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        pBackSub = createBackgroundSubtractorMOG2();
    else
        pBackSub = createBackgroundSubtractorKNN();

    // open input video
    VideoCapture capture(parser.get<String>("input"));
    if (!capture.isOpened()) {
        cerr << "Unable to open video" << endl;
        return -1;
    }

    // define fence polygon
    vector<Point> fence_points = { Point(30, 130), Point(255, 130), Point(255, 300), Point(30, 300) };
    vector<vector<Point>> fence_contours = { fence_points };

    // read frames from video
    Mat frame, fgMask, masked_fgMask, fgMask_thresh, subtraction, final;
    bool personInside = false;

    while (capture.read(frame)) {
        // apply background subtraction
        pBackSub->apply(frame, fgMask);

        // apply morphological operations to reduce noise
        medianBlur(fgMask, fgMask, 7);
        vector<vector<Point>> contours_vector;
        findContours(fgMask, contours_vector, RETR_TREE, CHAIN_APPROX_NONE);
        Mat contourImage(fgMask.size(), CV_8UC1, Scalar(0));
        for (unsigned short contour_index = 0; contour_index < contours_vector.size(); contour_index++) {
            drawContours(contourImage, contours_vector, contour_index, Scalar(255), -1); }
        subtraction = contourImage.clone();

        // apply fence mask
        masked_fgMask = Mat::zeros(contourImage.size(), contourImage.type());
        fillPoly(masked_fgMask, fence_contours, Scalar(255, 255, 255));
        bitwise_and(contourImage, masked_fgMask, contourImage);

        // threshold mask to separate foreground and background pixels
        threshold(contourImage, fgMask_thresh, 10, 255, THRESH_BINARY);

        // find contours of foreground blobs
        vector<vector<Point>> contours;
        findContours(fgMask_thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        final = frame.clone();

        bool personInside_temp = false;
        for (const auto& contour : contours) {
            Rect bbox = boundingRect(contour);
            if (bbox.area() > 2000 && bbox.y > fence_points[0].y) {
                rectangle(final, bbox, Scalar(0, 0, 255), 2);
                personInside_temp = true;
            }
        }
        personInside = personInside_temp;

        // draw red rectangle around invading area
        if (personInside)
            rectangle(final, Point(30, 130), Point(255, 300), Scalar(0, 0, 255), 2); // red
        else
            rectangle(final, Point(30, 130), Point(255, 300), Scalar(255, 0, 0), 2); // blue

        // show frames
        imshow ("final", final);
        imshow ("background subtraction", subtraction);

        // get the input from the keyboard
        int keyboard = waitKey(30);
        if (keyboard == 27) //esc key
            break;
    }
    return 0;
}
