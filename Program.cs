using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using Microsoft.ML;
using Microsoft.ML.Data;

public class HoaxData
{
    [LoadColumn(3)]
    public string Flag { get; set; } = null!;

    [LoadColumn(5)]
    public string PostText { get; set; } = null!;
}

public class HoaxPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedFlag { get; set; } = null!;

    public float[] Score { get; set; } = null!;
}

class Program
{
    private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "idn-turnbackhoax-2025.csv");
    private static readonly string _chartPath = Path.Combine(Environment.CurrentDirectory, "DistribusiLabel_QuickChart.png");

    static void Main(string[] args)
    {
        Console.WriteLine("Proyek Deteksi Hoax ML.NET Dimulai...");
        
        var context = new MLContext(seed: 0);

        Console.WriteLine($"Loading data from: {_dataPath}");
        var dataView = context.Data.LoadFromTextFile<HoaxData>(
            path: _dataPath,
            hasHeader: true,
            separatorChar: ';',
            allowQuoting: true
        );

        Console.WriteLine("Membuat visualisasi distribusi data...");
        var dataList = context.Data.CreateEnumerable<HoaxData>(dataView, reuseRowObject: false).ToList();
        var flagCounts = dataList
            .Where(x => !string.IsNullOrEmpty(x.Flag))
            .Select(x => {
                string cleanedFlag = x.Flag.ToUpper().Trim();
                if (cleanedFlag == "SATIR")
                {
                    cleanedFlag = "SATIRE";
                }
                x.Flag = cleanedFlag;
                return x;
            })
            .GroupBy(x => x.Flag)
            .Select(g => new { Flag = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count)
            .ToList();
        
        string labels = string.Join("','", flagCounts.Select(x => $"{x.Flag} ({x.Count})"));
        string data = string.Join(",", flagCounts.Select(x => x.Count));
        string jsonConfig = $@"
        {{
          type: 'bar',
          // backgroundColor: 'white', <-- HAPUS DARI SINI
          data: {{
            labels: ['{labels}'],
            datasets: [{{
              label: 'Jumlah Data',
              data: [{data}],
              backgroundColor: 'rgba(54, 162, 235, 0.5)',
              borderColor: 'rgba(54, 162, 235, 1)',
              borderWidth: 1
            }}]
          }},
          options: {{
            title: {{
              display: true,
              text: 'Distribusi Label Hoax (Data Imbalance)'
            }},
            legend: {{
              display: false
            }},
            scales: {{
              yAxes: [{{
                ticks: {{
                  beginAtZero: true
                }},
                scaleLabel: {{
                  display: true,
                  labelString: 'Jumlah Data'
                }}
              }}]
            }}
          }}
        }}";
        
        // FIX 2: Tambahin background-nya di URL, bukan di JSON
        string url = $"https://quickchart.io/chart?c={WebUtility.UrlEncode(jsonConfig)}&backgroundColor=white";

        try
        {
            using (var client = new HttpClient())
            {
                byte[] imageBytes = client.GetByteArrayAsync(url).Result;
                File.WriteAllBytes(_chartPath, imageBytes);
                
                Console.WriteLine($"\n==================================================");
                Console.WriteLine($"Chart gambar BERHASIL disimpan di: {_chartPath}");
                Console.WriteLine($"==================================================");
            }
        }
        catch (Exception e)
        {
            Console.WriteLine($"\nGAGAL DOWNLOAD GAMBAR CHART: {e.Message}");
        }

        var trainTestData = context.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 0);
        var trainData = trainTestData.TrainSet;
        var testData = trainTestData.TestSet; 

        Console.WriteLine("\nMembangun pipeline...");
        var pipeline =
            context.Transforms.Conversion.MapValueToKey(inputColumnName: "Flag", outputColumnName: "Label")
            .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "PostText", outputColumnName: "Features"))
            .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
            .Append(context.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

        Console.WriteLine("Mulai Training Model...");
        var model = pipeline.Fit(trainData);
        Console.WriteLine("Model selesai di-training.");

        Console.WriteLine("Mulai Evaluasi Model dengan data test...");
        var predictions = model.Transform(testData);
        var metrics = context.MulticlassClassification.Evaluate(predictions, "Label", "Score", "PredictedLabel");

        Console.WriteLine("==================================================");
        Console.WriteLine("       HASIL EVALUASI MODEL (METRICS)");
        Console.WriteLine("==================================================");
        Console.WriteLine($"  MicroAccuracy:    {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"  MacroAccuracy:    {metrics.MacroAccuracy:P2}");
        Console.WriteLine($"  LogLoss:          {metrics.LogLoss:F4}");
        Console.WriteLine("\n  (MicroAccuracy adalah Akurasi Keseluruhan)");
        Console.WriteLine("==================================================");

        Console.WriteLine("\nMembuat 1 contoh prediksi...");
        var predictionEngine = context.Model.CreatePredictionEngine<HoaxData, HoaxPrediction>(model);
        var sampleData = new HoaxData()
        {
            PostText = "Ini contoh kasus, ketidakadilan kpd konsumen. Mie Gacoan disegel , infonya disebabkan mengandung🐖 Babi ??"
        };
        var predictionResult = predictionEngine.Predict(sampleData);

        Console.WriteLine("\n==================================================");
        Console.WriteLine("         HASIL PREDIKSI 1 DATA CONTOH");
        Console.WriteLine("==================================================");
        Console.WriteLine($"  Teks Berita: {sampleData.PostText}");
        Console.WriteLine($"  Prediksi Flag: {predictionResult.PredictedFlag}");
        Console.WriteLine("==================================================");

        context.Model.Save(model, dataView.Schema, "hoax_detection_model.zip");
        Console.WriteLine("\nModel telah disimpan ke file: hoax_detection_model.zip");
        Console.WriteLine("Proses Selesai.");
    }
}