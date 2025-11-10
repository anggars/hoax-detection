using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

public class HoaxData
{
    [LoadColumn(3)]
    public string Flag { get; set; }

    [LoadColumn(5)]
    public string PostText { get; set; }
}

public class HoaxPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedFlag { get; set; }

    public float[] Score { get; set; }
}

class Program
{
    private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "idn-turnbackhoax-2025.csv");

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

        var trainTestData = context.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 0);
        var trainData = trainTestData.TrainSet;
        var testData = trainTestData.TestSet;

        Console.WriteLine("Membangun pipeline...");
        var pipeline =
            context.Transforms.Conversion.MapValueToKey(inputColumnName: "Flag", outputColumnName: "Label")
            .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "PostText", outputColumnName: "Features"))
            .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
            .Append(context.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

        Console.WriteLine("Mulai Training Model... (Ini mungkin butuh beberapa detik)");
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
        Console.WriteLine($"\n  (MicroAccuracy adalah Akurasi Keseluruhan)");
        Console.WriteLine("==================================================");
        Console.WriteLine("\nSCREENSHOT HASIL DI ATAS UNTUK DOKUMEN PERANCANGAN!");

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