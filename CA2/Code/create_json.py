import json

datasets = [
    "artificialNoAnomaly/art_daily_no_noise",
    "artificialNoAnomaly/art_daily_perfect_square_wave",
    "artificialNoAnomaly/art_daily_small_noise",
    "artificialNoAnomaly/art_flatline",
    "artificialNoAnomaly/art_noisy",
    "artificialWithAnomaly/art_daily_flatmiddle",
    "artificialWithAnomaly/art_daily_jumpsdown",
    "artificialWithAnomaly/art_daily_jumpsup",
    "artificialWithAnomaly/art_daily_nojump",
    "artificialWithAnomaly/art_increase_spike_density",
    "artificialWithAnomaly/art_load_balancer_spikes",
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533",
    "realAWSCloudwatch/ec2_cpu_utilization_77c1ca",
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd",
    "realAWSCloudwatch/ec2_cpu_utilization_c6585a",
    "realAWSCloudwatch/ec2_cpu_utilization_fe7f93",
    "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de",
    "realAWSCloudwatch/ec2_disk_write_bytes_c0d644",
    "realAWSCloudwatch/ec2_network_in_257a54",
    "realAWSCloudwatch/ec2_network_in_5abac7",
    "realAWSCloudwatch/elb_request_count_8c0756",
    "realAWSCloudwatch/grok_asg_anomaly",
    "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b",
    "realAdExchange/exchange-2_cpc_results",
    "realAdExchange/exchange-2_cpm_results",
    "realAdExchange/exchange-3_cpc_results",
    "realAdExchange/exchange-3_cpm_results",
    "realAdExchange/exchange-4_cpc_results",
    "realAdExchange/exchange-4_cpm_results",
    "realKnownCause/ambient_temperature_system_failure",
    "realKnownCause/cpu_utilization_asg_misconfiguration",
    "realKnownCause/ec2_request_latency_system_failure",
    "realKnownCause/machine_temperature_system_failure",
    "realKnownCause/nyc_taxi",
    "realKnownCause/rogue_agent_key_hold",
    "realKnownCause/rogue_agent_key_updown",
    "realTraffic/TravelTime_387",
    "realTraffic/TravelTime_451",
    "realTraffic/occupancy_6005",
    "realTraffic/occupancy_t4013",
    "realTraffic/speed_6005",
    "realTraffic/speed_7578",
    "realTraffic/speed_t4013",
    "realTweets/Twitter_volume_AAPL",
    "realTweets/Twitter_volume_AMZN",
    "realTweets/Twitter_volume_CRM",
    "realTweets/Twitter_volume_CVS",
    "realTweets/Twitter_volume_FB",
    "realTweets/Twitter_volume_GOOG",
    "realTweets/Twitter_volume_IBM",
    "realTweets/Twitter_volume_KO",
    "realTweets/Twitter_volume_PFE",
    "realTweets/Twitter_volume_UPS"
]

default_params = {
    "train_ratio": 0.4,
    "calib_ratio": 0.1,
    "window_size": 30,
    "epochs": 30,
    "lr": 0.001,
    "droupout": 0.3,
    "conf": 0.9
}

config_dict = {f"{name}.csv": default_params for name in datasets}

with open("params_supervised.json", "w") as f:
    json.dump(config_dict, f, indent=2)

print("JSON file created.")
