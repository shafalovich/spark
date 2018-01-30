import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.mllib.stat.test.ChiSqTestResult;
import org.apache.spark.mllib.stat.test.KolmogorovSmirnovTestResult;
import org.apache.spark.sql.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import static org.apache.spark.mllib.random.RandomRDDs.normalJavaRDD;

public class Task6 {

    public static final String TASK_FOLDER = "src/main/resources/task6";
    public static final String FILE_PATH = TASK_FOLDER + "/students.csv";
    public static final double MIN_SCORE = 0.0;
    public static final double MAX_SCORE = 10.0;
    public static final double INTERVAL = 0.1;
    public static final int STUDENTS_COUNT = 9999999;
    public static final double MEAN = (MAX_SCORE - MIN_SCORE) / 2;
    public static final double VARIANCE = Math.pow((MAX_SCORE - MIN_SCORE) / 6, 2);// M+-3S = [0, 10], i.e. 6S=10=>S=5/3 =>D=S^2=25/9
    public static final int INTERVALS_COUNT = (int)((MAX_SCORE - MIN_SCORE) / INTERVAL);
    public static int realStudentsCount;

    private static SparkSession spark;

    public static void main(String[] args) throws IOException {
        spark = createSparkSession();
        realStudentsCount = generateFile();
        System.out.println("Start reading...");
        Dataset<Row> studentsDS = spark.read().option("header", true).csv(FILE_PATH);

        long startTime = System.currentTimeMillis();
        testKolmogorovSmirnov(studentsDS);
        long ksEndTime = System.currentTimeMillis();
        System.out.println("Kolmogorov-Smirnov time: " + (ksEndTime - startTime));
        System.out.println();

        testChiSq(studentsDS);
        long endTime = System.currentTimeMillis();
        System.out.println("ChiSq time: " + (endTime - ksEndTime));
    }

    private static void testKolmogorovSmirnov(Dataset<Row> studentsDS){
        KolmogorovSmirnovTestResult result = Statistics.kolmogorovSmirnovTest(
                studentsDS.select("score").toJavaRDD().map(row -> Double.parseDouble(row.getString(0))).mapToDouble(d -> d),
                "norm",
                MEAN,
                Math.sqrt(VARIANCE));
        System.out.println(result);
    }

    private static void testChiSq(Dataset<Row> studentsDS){
        Column scoreColumn = new Column("score");
        Dataset<Row> scoreRanks = studentsDS.withColumn("score_rank", functions.floor(scoreColumn.divide(INTERVAL)))
                .groupBy("score_rank").count();
        final double [] realScoreFrequencies = new double[INTERVALS_COUNT];
        scoreRanks.collectAsList().forEach(row -> realScoreFrequencies[(int)row.getLong(0)] = (double) row.getLong(1));

        ChiSqTestResult independenceTestResult = Statistics.chiSqTest(Vectors.dense(realScoreFrequencies),calculateTheoreticalFreq());
        System.out.println(independenceTestResult);
    }

    private static Vector calculateTheoreticalFreq(){
        NormalDistribution normalDistribution = new NormalDistribution(MEAN, Math.sqrt(VARIANCE));
        double[] theoreticalFrequencies = new double[INTERVALS_COUNT];
        double x = MIN_SCORE;
        for (int i = 0; i < INTERVALS_COUNT; i++) {
            theoreticalFrequencies[i] =
                    realStudentsCount * (normalDistribution.cumulativeProbability(x+INTERVAL) - normalDistribution.cumulativeProbability(x));
            x += INTERVAL;
        }
        return Vectors.dense(theoreticalFrequencies);
    }

    private static int generateFile() throws IOException{
        File folder = new File(TASK_FOLDER);
        if (!folder.exists()) {
            folder.mkdirs();
        }

        FileWriter fileWriter = new FileWriter(FILE_PATH);
        PrintWriter printWriter = new PrintWriter(fileWriter);
        printWriter.println("id,first_name,last_name,score");
        JavaDoubleRDD standardNormal = normalJavaRDD(JavaSparkContext.fromSparkContext(spark.sparkContext()), STUDENTS_COUNT);
        List<Double> scores = standardNormal.mapToDouble(x -> MEAN + Math.sqrt(VARIANCE) * x).take(STUDENTS_COUNT);

        int fakeStudentsCount = 0;
        for (int i = 0; i < STUDENTS_COUNT; i++) {
            double score = scores.get(i);
            if (score < MIN_SCORE || score > MAX_SCORE){
                fakeStudentsCount++;
                continue;
            }
            printWriter.println(i +
                    ",Name" + i
                    + ",LastName" + i
                    + "," + scores.get(i)
            );
        }
        printWriter.close();
        return STUDENTS_COUNT - fakeStudentsCount;
    }

    private static SparkSession createSparkSession(){
        SparkSession spark = SparkSession
                .builder()
                .appName("Task6 Students' scores")
                .master("local[*]")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }
}
