# ----------------------------------------------------------------------
# Anish Rao
# 20066423
# Supervised LSTM Anomaly Detection on NAB Dataset
# Pure Conformal Prediction Classifier
# ----------------------------------------------------------------------

# import Pkg
# packages = ["BSON", "CSV", "DataFrames", "DelimitedFiles", 
# "Flux", "JSON", "Statistics"]
# for pkg in packages
#     Pkg.add(pkg)
# end
# Pkg.update()

using BSON: @save
using CSV
using DataFrames
using Dates
using DelimitedFiles
using Flux
using JSON
using Random
using Statistics

Random.seed!(86)
```
---------------------------------- PARAMETER LOADING ----------------------------------
```
function load_params(json_path::String)::Dict{String, Dict{String, Float64}}
    if isfile(json_path)
        return JSON.parsefile(json_path)
    else
        error("params_supervised.json not found at $json_path")
    end
end
```
---------------------------------- DATA LOADING ----------------------------------
```
function load_data_with_labels(filepath::String, labelpath::String)
    all_labels = JSON.parsefile(labelpath)
    dataset_key = joinpath(splitpath(filepath)[end-1:end]...)
    
    df = CSV.read(filepath, DataFrame)
    timestamps = DateTime.(df.timestamp, dateformat"yyyy-MM-dd HH:MM:SS")
    values = Float32.(df.value)

    anomaly_windows = get(all_labels, dataset_key, [])
    
    labels = Vector{Int}(undef, length(timestamps))
    for (i, ts) in enumerate(timestamps)
        labels[i] = any(DateTime(w[1], dateformat"yyyy-MM-dd HH:MM:SS.ssss") <= ts <= 
        DateTime(w[2], dateformat"yyyy-MM-dd HH:MM:SS.ssss") for w in anomaly_windows) ? 1 : 0
    end

    return timestamps, values, labels
end
```
---------------------------------- DATA NORMALIZATION and SPLITTING ----------------------------------
```
function split_and_normalize(values, labels, 
    train_ratio::Float64, calib_ratio::Float64)

    n = length(values)
    train_end = floor(Int, n * train_ratio)
    calib_end = floor(Int, n * (train_ratio + calib_ratio))

    train_raw = values[1:train_end]
    mean_train = mean(train_raw)
    std_train = std(train_raw)
    normal_values = (values .- mean_train) ./ std_train

    train_values = normal_values[1:train_end]
    train_labels = labels[1:train_end]

    calib_values = normal_values[train_end+1:calib_end]
    calib_labels = labels[train_end+1:calib_end]

    test_values = normal_values[calib_end+1:end]
    test_labels = labels[calib_end+1:end]

    return (;train_values, train_labels,
           calib_values, calib_labels,
           test_values, test_labels)
end

```
---------------------------------- DATA SEQUENCING ----------------------------------
```
function split_windows(values::Vector{Float32}, labels::Vector{Int}, window_size::Int)
    n_seq = length(values) - window_size
    x = Array{Float32}(undef, window_size, 1, n_seq)
    y = Array{Float32}(undef, 1, n_seq)
    
    for i in 1:n_seq
        x[:, 1, i] = values[i:(i + window_size - 1)]
        y[1, i] = labels[i + window_size - 1]
    end

    return x, y
end
```
---------------------------------- MODEL TRAINING ----------------------------------
```
function setup_model(window_size::Int, dropout::Float64)
    return Chain(
        LSTM(window_size => 64),
        Dropout(dropout),
        Dense(64 => 1),
        sigmoid
    )
end

function train_model(x_train::Array{Float32, 3}, y_train::Array{Float32,2};
    window_size::Int, epochs::Int, lr::Float64, dropout::Float64)
    
    model = setup_model(window_size, dropout)
    opt_state = Flux.setup(ADAMW(lr), model)

    for epoch in 1:epochs
        Flux.reset!(model)

        loss, grads = Flux.withgradient(model) do m
            y_pred = m(x_train)
            y_true = reshape(y_train, size(y_pred))
            Flux.logitbinarycrossentropy(y_pred, y_true)
        end

        Flux.update!(opt_state, model, grads[1])
        println("Epoch $epoch -- Loss: $(round(loss, digits=4))")
    end
    
    return model
end
```
---------------------------------- CONFORMAL CLASSIFICATION  ----------------------------------
```
function conformal_threshold(model, x_calib, y_calib, x_test, confidence::Float64)
    Flux.reset!(model)
    calib_prob = model(x_calib)
    residuals = vec(abs.(calib_prob .- y_calib))
    
    if any(isnan, residuals)
        return NaN, NaN, NaN
    end

    threshold = quantile(residuals, confidence)

    println("\nThreshold = $(round(threshold, digits=4))")

    Flux.reset!(model)
    test_prob = model(x_test)
    test_residuals = abs.(test_prob .- 0.0)
    predicted_anomalies = test_residuals .> threshold

    return predicted_anomalies, test_prob, threshold

end
```
---------------------------------- EVALUATION ----------------------------------
```
function calc_metrics(filename:: String, y_true::Vector{Int}, y_pred::BitVector)
    TP = sum((y_true .== 1) .& (y_pred .== true))
    FP = sum((y_true .== 0) .& (y_pred .== true))
    FN = sum((y_true .== 1) .& (y_pred .== false))

    precision = TP + FP == 0 ? 0.0 : TP / (TP + FP)
    recall = TP + FN == 0 ? 0.0 : TP / (TP + FN)
    f1 = precision + recall == 0 ? 0.0 : 2 * (precision * recall) / (precision + recall)

    println("\nEvaluation Metrics for $filename")
    println("TP = $TP | FP = $FP | FN = $FN")
    println("Precision: $(round(precision * 100, digits=2))%")
    println("Recall: $(round(recall * 100, digits=2))%")
    println("F1 Score: $(round(f1 * 100, digits=2))%")

    return (;precision, recall, f1, TP, FP, FN)
end
```
---------------------------------- SAVING METRICS ----------------------------------
```
function save_metrics(filename::String, metrics::NamedTuple)
    mkpath("results")
    results_file = joinpath("results", "supervised_metrics.csv")

    header = ["filename", "TP", "FP", "FN", "Precision (%)", "Recall (%)", "F1 Score (%)"]

    row = [
        filename, metrics.TP, metrics.FP, metrics.FN,
        round(metrics.precision * 100, digits=2),
        round(metrics.recall * 100, digits=2),
        round(metrics.f1 * 100, digits=2)
    ]

    if isfile(results_file)
        open(results_file, "a") do io
            writedlm(io, [row], ',')
        end
    else
        open(results_file, "w") do io
            writedlm(io, [header], ',')
            writedlm(io, [row], ',')
        end
    end
end
```
---------------------------------- IMPLEMENTATION ----------------------------------
```
data_root = "data"
label_path = "labels/combined_windows.json"
param_path = "params_supervised.json"

params = load_params(param_path)

all_files = sort(collect(keys(params)))

for filename in all_files
    println("\nProcessing: $filename")

    file_path = joinpath(data_root, filename)

    # Load data and parameters
    timestamps, values, labels = load_data_with_labels(file_path, label_path)
    config = params[filename]
    train_ratio = config["train_ratio"]
    calib_ratio = config["calib_ratio"]
    window_size = Int(config["window_size"])
    epochs = Int(config["epochs"])
    lr = Float64(config["lr"])
    dropout = Float64(config["dropout"])
    conf = Float64(config["conf"])

    # Split and normalize data
    clean_data = split_and_normalize(values, labels, train_ratio, calib_ratio)

    # Create sequences
    x_train, y_train = split_windows(clean_data.train_values, clean_data.train_labels, window_size)
    x_calib, y_calib = split_windows(clean_data.calib_values, clean_data.calib_labels, window_size)
    x_test, y_test = split_windows(clean_data.test_values, clean_data.test_labels, window_size)

    # Train model
    model = train_model(
        x_train, y_train;
        window_size=window_size,
        epochs=epochs,
        lr=lr,
        dropout=dropout
    )

    # Classification
    y_pred_class, test_prob, threshold = conformal_threshold(
        model, x_calib, y_calib, x_test, conf)
    
    if !isfinite(threshold)
        save_metrics(filename, (;TP=0, FP=0, FN=0, precision=0.0, recall=0.0, f1=0.0, mse=0.0, mae=0.0))
        continue
    end

    # Evaluation
    y_true = Int.(vec(y_test))
    y_pred = vec(y_pred_class)
    metrics = calc_metrics(filename, y_true, y_pred)

    # Save metrics
    save_metrics(filename, metrics)
end
println("\nAll files processed successfully")