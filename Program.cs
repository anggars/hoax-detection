// Import library yang kita butuhkan
using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

// Ini adalah "cetakan" untuk INPUT data kita (data dari CSV)
public class HoaxData
{
    // [LoadColumn(3)] artinya: ambil data dari kolom ke-3 (indeks 3) di CSV
    // Kolom ke-3 adalah 'flag' (SALAH, MISLEADING, dll)
    [LoadColumn(3)]
    public string Flag { get; set; }

    // [LoadColumn(5)] artinya: ambil data dari kolom ke-5 (indeks 5) di CSV
    // Kolom ke-5 adalah 'post_text' (isi teks beritanya)
    [LoadColumn(5)]
    public string PostText { get; set; }
}

// Ini adalah "cetakan" untuk OUTPUT data kita (hasil prediksi model)
public class HoaxPrediction
{
    // [ColumnName("PredictedLabel")] adalah nama default ML.NET untuk hasil prediksi
    [ColumnName("PredictedLabel")]
    public string PredictedFlag { get; set; }

    // Kita juga akan minta skor probabilitasnya
    public float[] Score { get; set; }
}

// Ini adalah program utama kita
class Program
{
    static void Main(string[] args)
    {
        // Program kita akan mulai di sini.
        // Untuk sekarang, kita kosongkan dulu.
        Console.WriteLine("Proyek Deteksi Hoax ML.NET Dimulai...");

        // Nanti kita akan panggil fungsi training model di sini
    }
}