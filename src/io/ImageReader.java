package io;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

public class ImageReader {

    public static void main(String[] args) throws Exception {
        readImage("saves/digit/images/six.png");
    }

    public static void readImage(String path) throws Exception {
        BufferedImage img = ImageIO.read(new File(path));
        WritableRaster raster = img.getRaster();
        System.out.println(raster.getWidth());
        int[] color = { 30, 10, 200 };
        raster.setPixel(3, 3, color);
        ImageIO.write(img, "png", new File("saves/digit/images/six2.png"));
    }

    public static void writeImage(int width, int height, int[] pixels, String savePath) throws IOException {
        BufferedImage img = ImageIO.read(new File("saves/digit/images/six.png"));
        System.out.println(pixels.length);
        WritableRaster raster = img.getRaster();
        int[] pixel = { 255 - pixels[0], 255 - pixels[0], 255 - pixels[0] };
        raster.setPixel(4, 5, pixel);
        try {
            ImageIO.write(img, "png", new File("saves/digit/images/read1.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
