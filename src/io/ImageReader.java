package io;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

public class ImageReader {


	public static double[] readImage(String path) throws Exception {
		BufferedImage img = ImageIO.read(new File(path));
		double[] output = new double[img.getWidth() * img.getHeight()];
		for (int x = 0; x < img.getWidth(); x++)
			for (int y = 0; y < img.getHeight(); y++) {
				output[y * img.getWidth() + x] = toBlackAndWhite(img.getRGB(x, y));
			}
		return output;
	}

	public static void saveImage(int width, int height, int[] pixels, String savePath) throws IOException {
		BufferedImage img = ImageIO.read(new File("saves/digit/images/sample.png"));
		WritableRaster raster = img.getRaster();
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int index = width * y + x;
				int[] pixel = { 255 - pixels[index], 255 - pixels[index], 255 - pixels[index] };
				raster.setPixel(x, y, pixel);
			}
		}

		try {
			ImageIO.write(img, "png", new File(savePath));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static double toBlackAndWhite(int rgb) {
		int[] pixel = new int[3];
		pixel[0] = rgb & 0xff;
		pixel[1] = (rgb & 0xff00) >> 8;
		pixel[2] = (rgb & 0xff0000) >> 16;
		return toBlackAndWhite(pixel);
	}

	private static double toBlackAndWhite(int[] rgb) {
		double average = 0;
		for (int i = 0; i < rgb.length; i++)
			average += 255 - rgb[i];
		return average / (255 * rgb.length);
	}

	private static int[] toRGB(double blackAndWhite) {
		int[] output = new int[3];
		for (int i = 0; i < output.length; i++)
			output[i] = (int) (blackAndWhite * 255);
		return output;
	}
}
