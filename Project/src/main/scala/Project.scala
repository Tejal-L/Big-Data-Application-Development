import org.apache.spark.SparkContext 
import org.apache.spark.SparkContext._ 
import org.apache.spark.SparkConf 
import scala.collection.mutable.ArrayBuffer  
import org.apache.spark.sql.SQLContext 
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans
import scala.xml._
import java.nio.file.Paths


object Project {

    def main (args: Array[String]) {
        
        if (args.length < 1) {
            System.err.println("Usage: Project <path to directory>")
            System.exit(1)
        }

        System.out.println("Starting Spark Context...")
        val sc = new SparkContext()

        val sqlContext = new SQLContext(sc)  
        import sqlContext._  
        import sqlContext.implicits._ 

        val directory = args(0)
        System.out.println("Cleaning Data ....")

        // Clean School data
        val schoolRdd = sc.wholeTextFiles(Paths.get(directory, "school.xml").toString).
                           flatMap(record => (XML.loadString(record._2) \\ "response" \\ "row" \\ "row" \\ "location_1").toIterator)

        val schoolLocations = schoolRdd.map(r => Vectors.dense((r \ "@latitude").toString.toDouble, (r \ "@longitude").toString.toDouble)).cache()

        // Clean Subway Data
        val subwayDF = sqlContext.read.format("com.databricks.spark.csv").
                                  option("header", "true").
                                  load(Paths.get(directory, "subway.csv").toString)

        val subwayLocations = subwayDF.select("the_geom").rdd.
                                       map(_.toString.split(" ").slice(1, 3).map(_.replaceAll("[\\[\\]\\(\\)]", ""))).
                                       map(loc => Vectors.dense(loc(1).toDouble, loc(0).toDouble)).cache()


        // Clean Crime data
        val crimeLocations = sqlContext.read.format("com.databricks.spark.csv").
                                        option("header", "true").
                                        load(Paths.get(directory, "crime.csv").toString).
                                        filter($"LAW_CAT_CD" === "FELONY").
                                        filter($"CMPLNT_FR_DT".rlike("2016")).
                                        filter($"OFNS_DESC".contains("MURDER") || $"OFNS_DESC".contains("ROBBERY")).
                                        select("Latitude", "Longitude").rdd.
                                        map(r => (r(0).toString, r(1).toString)).
                                        filter(r => r._1.nonEmpty && r._2.nonEmpty).
                                        map(r => Vectors.dense(r._1.toDouble, r._2.toDouble)).cache()


        // KMeans on Crime Data
        val crimeModel = KMeans.train(crimeLocations, 500, 20)
        val crimeClusterScore = crimeModel.predict(crimeLocations).
                                           map(r => (r, 1)).
                                           reduceByKey(_ + _).
                                           map(r => r._2)


        sc.parallelize(crimeModel.clusterCenters.zip(crimeClusterScore.collect())).
                                                 map(r => r._1(0).toString + "," + r._1(1).toString + "," + r._2).
                                                 coalesce(1).
                                                 saveAsTextFile("crimeCluster")

        val subwayModel = KMeans.train(subwayLocations, 200, 2)
        val subwayClusterScore = subwayModel.predict(subwayLocations).
                                             map(r => (r, 1)).
                                             reduceByKey(_ + _).
                                             map(r => r._2)


        sc.parallelize(subwayModel.clusterCenters.zip(subwayClusterScore.collect())).
                                                  map(r => r._1(0).toString + "," + r._1(1).toString + "," + r._2).
                                                  coalesce(1).
                                                  saveAsTextFile("subwayCluster")


        val schoolModel = KMeans.train(schoolLocations, 500, 2)
        val schoolClusterScore = schoolModel.predict(schoolLocations).
                                             map(r => (r, 1)).
                                             reduceByKey(_ + _).
                                             map(r => r._2)


        sc.parallelize(schoolModel.clusterCenters.zip(schoolClusterScore.collect())).
                                                  map(r => r._1(0).toString + "," + r._1(1).toString + "," + r._2).
                                                  coalesce(1).
                                                  saveAsTextFile("schoolCluster")


        // Union 
        val unionData = subwayLocations.union(schoolLocations).cache()
        val unionModel = KMeans.train(unionData, 500, 20)

        val schoolCluster = unionModel.predict(schoolLocations).map(r => (r, 1))
        val subwayCluster = unionModel.predict(subwayLocations).map(r => (r, 2))

        val unionCluster = schoolCluster.union(subwayCluster)

        // Remove Clusters which has only school or subway in it
        val filterClusterCenters = unionCluster.groupBy(_._1).
                                                map(r => (r._1, r._2.map(_._2).toSet.size, r._2.map(_._2).size)).
                                                filter(r => r._2 == 2).
                                                map(r => (unionModel.clusterCenters(r._1), r._3))

        filterClusterCenters.coalesce(1).map(r => r._1(0).toString + "," + r._1(1).toString + "," + r._2.toString).
                                         saveAsTextFile("schoolSubwayCluster")


        System.out.println("Stopping Spark Context...")
        sc.stop()
    }

}

