package nails.jonajo;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    static Bitmap imageBitmap;
    public ImageView Iv;
    public ImageView Iv2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }

        Iv = (ImageView) findViewById(R.id.imageView);
        Iv2 = (ImageView) findViewById(R.id.imageView2);

        ////int drawableID = this.getResources().getIdentifier("finger", "drawable", getPackageName());
        imageBitmap = BitmapFactory.decodeResource(this.getResources(), R.drawable.test_4);
        //Iv.setImageBitmap(imageBitmap);

        ///////////////////////////////////////////////////////////////////////////////////
        /////////////////////// Testing Watershed /////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////
        Mat mat = new Mat();
        Bitmap bmp32 = imageBitmap.copy(Bitmap.Config.RGB_565, true);
        //bmp32.setConfig(Bitmap.Config.RGB_565);
        Utils.bitmapToMat(bmp32, mat);
        //Log.i("tipo mat", String.valueOf(mat.type()));
        Mat mat8UC = new Mat(mat.size(), CvType.CV_8UC(3));
        Imgproc.cvtColor(mat, mat8UC, Imgproc.COLOR_RGBA2RGB);

        //mat = ImageProcessing.Watershed(Mat.ones(640, 640, CvType.CV_8UC(3)));
        mat8UC = ImageProcessing.Watershed(mat8UC);
        Iv2.setImageBitmap(ImageProcessing.convertMatToBitMap(mat8UC));

        //try {
            //mat = Utils.loadResource(this, R.drawable.finger, CvType.CV_8UC4);
        //} catch (IOException e) {
            //e.printStackTrace();
        //}

        //mat = Imgcodecs.imread(getResources().getDrawable(R.drawable.finger).toString());
        ///////////////////////////////////////////////////////////////////////////////////

    }

}
