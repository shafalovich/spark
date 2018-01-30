import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.sum;

public class Task5 {

    public static final String CSV_FILE = "src/main/resources/task5/call_log.csv";
    public static final String PARQUET_FILE = "src/main/resources/task5/call_log.parquet";
    public static final int LOG_ITEMS_COUNT = 4999999;
    public static final int SUBSCRIBERS_COUNT = 15000;
    public static final int TOP_N_COUNT = 5;

    private static SparkSession spark;

    public static void main(String[] args) throws IOException {
        spark = createSparkSession();

        createFiles();

        showTopSubscribers(true);
        showTopSubscribers(false);

        spark.close();
    }

    private static void showTopSubscribers(boolean isCSV){
        System.out.println("Starting calculation...");
        long startTime = System.currentTimeMillis();
        Dataset<Row> callLogs = isCSV ?
                spark.read().option("header", true).csv(CSV_FILE)
                : spark.read().parquet(PARQUET_FILE);
        long readTime = System.currentTimeMillis();
        Dataset<Row> groupBySubscriberOrderByTotal = callLogs.groupBy("from").agg(sum("duration").alias("total_duration")).orderBy(desc("total_duration"));
        groupBySubscriberOrderByTotal.show(TOP_N_COUNT);
        long endTime = System.currentTimeMillis();
        System.out.println(String.format("%s: Read time: %d; Total time: %d",isCSV?"CSV":"PARQUET",readTime-startTime, endTime - startTime));
        spark.sqlContext().clearCache();
    }

    private static void createFiles() throws IOException {
        FileWriter fileWriter = new FileWriter(CSV_FILE);
        PrintWriter printWriter = new PrintWriter(fileWriter);
        printWriter.println("start_timest,from,to,duration,region,position");
        for (int i = 0; i < LOG_ITEMS_COUNT; i++) {
            printWriter.println(i + ","
                    + Math.round(Math.random() * (SUBSCRIBERS_COUNT-1))
                    + "," + Math.round(Math.random() * (SUBSCRIBERS_COUNT - 1))
                    + "," + Math.round(Math.random() * 30)
                    + ",region" + Math.round(Math.random() * 16)
                    + ",position" + Math.round(Math.random() * 1000)
            );
        }
        printWriter.close();

        File folder = new File(PARQUET_FILE);
        if (folder.exists()) {
            String[] entries = folder.list();
            for(String s: entries){
                File currentFile = new File(folder.getPath(),s);
                currentFile.delete();
            }
            folder.delete();
        }
        Dataset<Row> data = spark.read().option("header", true).csv(CSV_FILE);
        data.write().parquet(PARQUET_FILE);
        spark.sqlContext().clearCache();
    }


    private static SparkSession createSparkSession(){
        SparkSession spark = SparkSession
                .builder()
                .appName("Task5 Top Subscribers")
                .master("local[*]")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }

}
