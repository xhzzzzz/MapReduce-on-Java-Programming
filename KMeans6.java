package com.kmean.houzexu;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

public class KMeans {

  public static class PointsMapper extends Mapper<LongWritable, Text, Text, Text> {

    private List<Point> centers = new ArrayList<>();

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
      super.setup(context);
      Configuration conf = context.getConfiguration();
      Path centroidsPath = new Path(conf.get("centroid.path"));
      FileSystem fs = FileSystem.get(conf);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroidsPath, conf);
      Text key = new Text();
      IntWritable value = new IntWritable();
      while (reader.next(key, value)) {
        centers.add(new Point(key.toString()));
      }
      reader.close();
    }

    @Override
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      Point point = new Point(value.toString());
      double minDistance = Double.MAX_VALUE;
      Point closestCenter = null;
      for (Point center : centers) {
        double distance = center.distanceTo(point);
        if (distance < minDistance) {
          minDistance = distance;
          closestCenter = center;
        }
      }
      context.write(new Text(closestCenter.toString()), new Text(point.toString()));
    }
  }

  public static class PointsReducer extends Reducer<Text, Text, Text, Text> {

    private List<Point> newCentroids = new ArrayList<>();

    @Override
    public void reduce(Text key, Iterable<Text> values, Context context)
        throws IOException, InterruptedException {
      List<Point> points = new ArrayList<>();
      for (Text value : values) {
        points.add(new Point(value.toString()));
      }
      Point newCentroid = calculateNewCentroid(points);
      newCentroids.add(newCentroid);
      
      context.write(key, new Text(newCentroid.toString()));
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
      Configuration conf = new Configuration();

      Path centroidsPath = new Path("centroid/cen.seq");
      conf.set("centroid.path", centroidsPath.toString());
      FileSystem fs = FileSystem.get(conf);
      
      if(fs.exists(centroidsPath)) {
    	  fs.delete(centroidsPath, true);
      }
      final SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, centroidsPath, Text.class, IntWritable.class);
      for (int i = 0; i < newCentroids.size(); i++) {
        Point centroid = newCentroids.get(i);
        writer.append(new Text(centroid.toString()), new IntWritable(i));
      }
      writer.close();
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Path centroidPath = new Path("centroid/cen.seq");
    conf.set("centroid.path", centroidPath.toString());
    FileSystem fs = FileSystem.get(conf);
    if (fs.exists(centroidPath)) {
      fs.delete(centroidPath, true);
    }
    final SequenceFile.Writer centerWriter = SequenceFile.createWriter(fs, conf, centroidPath, Text.class,
        IntWritable.class);
    final IntWritable value = new IntWritable(0);
    centerWriter.append(new Text("50.197031637442876,32.94048164287042"), value);
    centerWriter.append(new Text("43.407412339767056,6.541037020010927"), value);
    centerWriter.append(new Text("1.7885358732482017,19.666057053079573"), value);
    centerWriter.append(new Text("32.6358540480337,4.03843047564191"), value);
    centerWriter.append(new Text("32.4556754322344, 4.3456655433455"), value);
    centerWriter.append(new Text("30.7898763567283, 2.7898735268398"), value);
    centerWriter.close();

    int iteration = 0;
    while (iteration < 10) {
      Job job = Job.getInstance(conf, "KMeans");
      job.setJarByClass(KMeans.class);
      job.setMapperClass(PointsMapper.class);
//      job.setCombinerClass(PointsReducer.class);
      job.setReducerClass(PointsReducer.class);
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(Text.class);
     
      FileInputFormat.addInputPath(job, new Path(args[0])); // Set your input path
      FileOutputFormat.setOutputPath(job, new Path(args[1] + '_' + iteration)); // Set your output path
      job.waitForCompletion(true);
      iteration++;
    }

    // Read the final centroid file from HDFS and print the centroids (final result)
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroidPath, conf);
    Text key = new Text();
    IntWritable centroidId = new IntWritable();
    while (reader.next(key, centroidId)) {
      System.out.println("Centroid " + centroidId.get() + ": " + key.toString());
    }
    reader.close();
  }

  public static class Point {
    private double x;
    private double y;

    public Point(double x, double y) {
      this.x = x;
      this.y = y;
    }

    public Point(String pointString) {
      String[] parts = pointString.split(",");
      this.x = Double.parseDouble(parts[0]);
      this.y = Double.parseDouble(parts[1]);
    }

    public double getX() {
      return x;
    }

    public double getY() {
      return y;
    }

    public double distanceTo(Point other) {
      double dx = this.x - other.x;
      double dy = this.y - other.y;
      return Math.sqrt(dx * dx + dy * dy);
    }

    @Override
    public String toString() {
      return x + "," + y;
    }
  }

  // Helper method to calculate the new centroid based on a list of points
  private static Point calculateNewCentroid(List<Point> points) {
    double sumX = 0.0;
    double sumY = 0.0;
    for (Point point : points) {
      sumX += point.getX();
      sumY += point.getY();
    }
    double avgX = sumX / points.size();
    double avgY = sumY / points.size();
    return new Point(avgX, avgY);
  }
}