import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class ImageProcessing {
    // class contains image processing algorithms

    /**
     * This function implements image segmentation to img by using the watershed algorithm.
     *
     * @param img     //The image in RGB format to which the algorithm will be applied.
     * @return //The image obtained with the watershed algorithm.
     *          In this image the sure background of img has been identified and colored black.
     *          The rest of the image remains the same.
     */
    public static Mat Watershed(Mat img){
        //watershed algorithm

        Mat gray = new Mat();
        Mat thresh = new Mat();
        double ret;
        Mat  kernel;
        Mat opening = new Mat();
        Mat sure_bg = new Mat();
        Point anchor = new Point(-1, -1); // default value
        Mat dist_transform = new Mat();
        Mat sure_fg = new Mat();
        Mat unknown = new Mat();
        Mat markers = new Mat();
        Scalar sc = new Scalar(-1);

        Imgproc.cvtColor(img, gray, Imgproc.COLOR_RGB2GRAY);
        ret = Imgproc.threshold(gray, thresh, 0 ,255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        // remove noise
        kernel = Mat.ones(3, 3, CvType.CV_8UC(1));
        Imgproc.morphologyEx(thresh, opening, Imgproc.MORPH_OPEN, kernel, anchor, 1);

        // Find the sure background region
        Imgproc.dilate(opening, sure_bg, kernel, anchor, 8);

        // Find the sure foreground region
        Imgproc.distanceTransform(opening, dist_transform,Imgproc.DIST_L2, 5);
        Core.MinMaxLocResult mmr= Core.minMaxLoc(dist_transform);
        ret = Imgproc.threshold(dist_transform, sure_fg, 0.7 * mmr.maxVal ,255, 0);
        sure_fg.assignTo(sure_fg, CvType.CV_8UC(sure_fg.channels()));
        //Log.i("value", String.valueOf(0.7 * mmr.maxVal));

        // Find the unknown region
        Core.subtract(sure_bg, sure_fg, unknown);

        // Label the foreground objects
        ret = Imgproc.connectedComponents(sure_fg, markers);

        // Add one to all labels so that sure background is not 0, but 1
        Core.subtract(markers, sc, markers);
        //for(int i=0; i <markers.rows(); i++){

          //  for(int j=0; j < markers.cols(); j++){
            //    markers.put(i, j, markers.get(i, j)[0] + 1);
            //}
        //}

        // Label the unknown region as 0
        for(int i=0; i <markers.rows(); i++){

            for(int j=0; j < markers.cols(); j++){
                if(unknown.get(i, j)[0] == 255){
                    markers.put(i, j, 0);
                }
            }
        }

        Imgproc.watershed(img, markers);

        for(int i=0; i <markers.rows(); i++){

            for(int j=0; j < markers.cols(); j++){
                if(markers.get(i, j)[0] == 1){
                    img.put(i, j, 0,0,0);
                }
            }
        }

        return img;
    }

    /**
     * This function converts images in RGB format from Mat to BitMap
     *
     * @param input     //The image in RGB format to be converted from Mat to BitMap
     * @return //The image converted to BitMap
     */
    public static Bitmap convertMatToBitMap(Mat input){
        Bitmap bmp = null;
        //Mat rgb = new Mat();
        //Imgproc.cvtColor(input, rgb, Imgproc.COLOR_BGR2RGB);

        Mat rgb = input;

        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        return bmp;
    }
}
