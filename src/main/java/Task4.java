import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.sum;

public class Task4 {

    public static final String TASK_FOLDER = "src/main/resources/task4";
    public static final String ORDER_ITEMS_FILE = TASK_FOLDER + "/order_items.csv";
    public static final String PRODUCTS_FILE = TASK_FOLDER + "/products.csv";
    public static final int PRODUCTS_COUNT = 5000;
    public static final int CATEGORIES_COUNT = 20;
    public static final int ORDER_ITEMS_COUNT = 500000;
    public static final int TOP_N_PRODUCTS = 3;

    public static void main(String[] args) throws IOException {
        createFiles();
        SparkSession spark = createSparkSession();

        System.out.println("Starting calculation...");
        long startTime = System.currentTimeMillis();
        Dataset<Row> orderItems = spark.read().option("header", true).csv(ORDER_ITEMS_FILE);
        Dataset<Row> products = spark.read().option("header", true).csv(PRODUCTS_FILE);

        Column orderItemCost = new Column("cost");
        Column orderItemsQuantity = new Column("qty");
        Column productPricePerOrder = orderItemCost.multiply(orderItemsQuantity);

        Dataset<Row> productTotals = orderItems.withColumn("total_price", productPricePerOrder).groupBy("product_id").agg(sum("total_price").alias("product_total"));
        //productTotals.show();

        Dataset<Row> productOrders = products.join(productTotals, "product_id");
        //productOrders.show();

        WindowSpec categoryPartitionsSortedByPrice = Window.partitionBy("category_id").orderBy(desc("product_total"));
        Dataset<Row> topRevenueProducts = productOrders.withColumn("rank", functions.rank().over(categoryPartitionsSortedByPrice)).where("rank <= " + TOP_N_PRODUCTS);

        long endTime = System.currentTimeMillis();
        topRevenueProducts.show(TOP_N_PRODUCTS * CATEGORIES_COUNT + 2);
        System.out.println("Total time: " + (endTime - startTime));

        spark.close();
    }

    private static void createFiles() throws IOException {
        File folder = new File(TASK_FOLDER);
        if (!folder.exists()) {
            folder.mkdirs();
        }

        FileWriter fileWriter = new FileWriter(ORDER_ITEMS_FILE);
        PrintWriter printWriter = new PrintWriter(fileWriter);
        printWriter.println("item_id,order_id,product_id,qty,promotion_id,cost");
        for (int i = 0; i < ORDER_ITEMS_COUNT; i++) {
            printWriter.println(i + ","
                    + Math.round(Math.random() * 300)
                    + "," + Math.round(Math.random() * (PRODUCTS_COUNT - 1))
                    + "," + Math.round(Math.random() * 10)
                    + "," + Math.round(Math.random() * 3)
                    + "," + Math.round(Math.random() * 10)
            );
        }
        printWriter.close();

        fileWriter = new FileWriter(PRODUCTS_FILE);
        printWriter = new PrintWriter(fileWriter);
        printWriter.println("product_id,category_id");
        for (int i = 0; i < PRODUCTS_COUNT; i++) {
            printWriter.println(i + "," + Math.round(Math.random() * (CATEGORIES_COUNT - 1)));
        }
        printWriter.close();
    }

    private static SparkSession createSparkSession(){
        SparkSession spark = SparkSession
                .builder()
                .appName("Task4 Top products")
                .master("local[*]")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }
}
