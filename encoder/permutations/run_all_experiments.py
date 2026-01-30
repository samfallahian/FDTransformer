#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master script to run all 10 autoencoder permutation experiments
Runs experiments sequentially or with limited parallelism and compares results
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration - EDIT THESE AS NEEDED
NUMEPOCHS = 100          # Number of epochs to train each model
THREAD_COUNT = 2         # Number of models to train simultaneously
GLOBAL_PERCENTAGE = 10   # Percentage of data to use (10 = 10%, 100 = all data)
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

# Resolve project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = [
    {
        'name': 'Model_01_Standard_VAE',
        'module': 'encoder.permutations.model_01_standard_vae',
        'class': 'StandardVAE',
        'description': 'Standard VAE with KL divergence'
    },
    {
        'name': 'Model_02_Beta_VAE',
        'module': 'encoder.permutations.model_02_beta_vae',
        'class': 'BetaVAE',
        'description': 'β-VAE with adjustable β=4.0'
    },
    {
        'name': 'Model_03_Sparse_AE',
        'module': 'encoder.permutations.model_03_sparse_ae',
        'class': 'SparseAE',
        'description': 'Sparse AE with L1 regularization'
    },
    {
        'name': 'Model_04_Contractive_AE',
        'module': 'encoder.permutations.model_04_contractive_ae',
        'class': 'ContractiveAE',
        'description': 'Contractive AE penalizing Jacobian'
    },
    {
        'name': 'Model_05_Denoising_AE',
        'module': 'encoder.permutations.model_05_denoising_ae',
        'class': 'DenoisingAE',
        'description': 'Denoising AE with Gaussian noise'
    },
    {
        'name': 'Model_06_Adversarial_AE',
        'module': 'encoder.permutations.model_06_adversarial_ae',
        'class': 'AdversarialAE',
        'description': 'Adversarial AE with discriminator'
    },
    {
        'name': 'Model_07_VQ_VAE',
        'module': 'encoder.permutations.model_07_vq_vae',
        'class': 'VQVAE',
        'description': 'Vector Quantized VAE'
    },
    {
        'name': 'Model_08_Deep_AE',
        'module': 'encoder.permutations.model_08_deep_ae',
        'class': 'DeepAE',
        'description': 'Deep AE with 6+ hidden layers'
    },
    {
        'name': 'Model_09_Residual_AE',
        'module': 'encoder.permutations.model_09_residual_ae',
        'class': 'ResidualAE',
        'description': 'Residual AE with skip connections'
    },
    {
        'name': 'Model_10_Mixture_AE',
        'module': 'encoder.permutations.model_10_mixture_ae',
        'class': 'MixtureDensityAE',
        'description': 'Mixture Density AE'
    }
]


def train_single_model(model_config):
    """
    Train a single model configuration

    Args:
        model_config: Dictionary with model configuration

    Returns:
        Dictionary with results
    """
    import importlib
    from encoder.permutations.train_permutation import main as train_main

    model_name = model_config['name']
    logger.info(f"Starting {model_name}: {model_config['description']}")

    start_time = time.time()

    try:
        # Create checkpoint directory for this model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(script_dir, 'checkpoints', model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Dynamically import the model class
        module = importlib.import_module(model_config['module'])
        model_class = getattr(module, model_config['class'])

        # Train the model
        results = train_main(
            model_class=model_class,
            model_name=model_name,
            num_epochs=NUMEPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            sample_percentage=GLOBAL_PERCENTAGE,
            save_dir=checkpoint_dir
        )

        elapsed_time = time.time() - start_time

        return {
            'name': model_name,
            'description': model_config['description'],
            'status': 'success',
            'final_train_rmse': results['final_train_rmse'],
            'final_val_rmse': results['final_val_rmse'],
            'train_rmse_history': results['train_rmse'],
            'val_rmse_history': results['val_rmse'],
            'training_time_seconds': elapsed_time,
            'num_epochs': NUMEPOCHS
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error training {model_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

        return {
            'name': model_name,
            'description': model_config['description'],
            'status': 'failed',
            'error': str(e),
            'training_time_seconds': elapsed_time
        }


def run_experiments_parallel():
    """Run all experiments with parallelism limited by THREAD_COUNT"""
    logger.info("="*80)
    logger.info(f"Starting all experiments")
    logger.info(f"Configuration: NUMEPOCHS={NUMEPOCHS}, THREAD_COUNT={THREAD_COUNT}")
    logger.info(f"Data Sampling: {GLOBAL_PERCENTAGE}% of full dataset")
    logger.info(f"Batch Size={BATCH_SIZE}, Learning Rate={LEARNING_RATE}")
    logger.info("="*80)

    all_results = []
    overall_start = time.time()

    # Run with limited parallelism
    with ProcessPoolExecutor(max_workers=THREAD_COUNT) as executor:
        # Submit all jobs
        future_to_model = {
            executor.submit(train_single_model, model_config): model_config
            for model_config in MODELS
        }

        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_config = future_to_model[future]
            try:
                result = future.result()
                all_results.append(result)

                if result['status'] == 'success':
                    logger.info(f"✓ {result['name']} completed: Val RMSE = {result['final_val_rmse']:.6f}")
                else:
                    logger.error(f"✗ {result['name']} failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Exception retrieving result for {model_config['name']}: {str(e)}")

    overall_time = time.time() - overall_start

    # Sort results by name for consistent ordering
    all_results.sort(key=lambda x: x['name'])

    # Print summary
    print_summary(all_results, overall_time)

    # Save results
    save_results(all_results, overall_time)

    return all_results


def run_experiments_sequential():
    """Run all experiments sequentially (for debugging or when THREAD_COUNT=1)"""
    logger.info("="*80)
    logger.info(f"Starting all experiments (SEQUENTIAL)")
    logger.info(f"Configuration: NUMEPOCHS={NUMEPOCHS}")
    logger.info(f"Data Sampling: {GLOBAL_PERCENTAGE}% of full dataset")
    logger.info(f"Batch Size={BATCH_SIZE}, Learning Rate={LEARNING_RATE}")
    logger.info("="*80)

    all_results = []
    overall_start = time.time()

    for i, model_config in enumerate(MODELS, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Training model {i}/{len(MODELS)}: {model_config['name']}")
        logger.info(f"{'='*80}")

        result = train_single_model(model_config)
        all_results.append(result)

        if result['status'] == 'success':
            logger.info(f"✓ {result['name']} completed: Val RMSE = {result['final_val_rmse']:.6f}")
        else:
            logger.error(f"✗ {result['name']} failed: {result.get('error', 'Unknown error')}")

    overall_time = time.time() - overall_start

    # Print summary
    print_summary(all_results, overall_time)

    # Save results
    save_results(all_results, overall_time)

    return all_results


def print_summary(results, overall_time):
    """Print a summary table of all results"""
    print("\n")
    print("="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print(f"Total time: {overall_time/60:.2f} minutes")
    print()

    # Separate successful and failed
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    if successful:
        # Sort by validation RMSE (best first)
        successful.sort(key=lambda x: x['final_val_rmse'])

        print(f"Successful Models ({len(successful)}):")
        print("-"*100)
        print(f"{'Rank':<6} {'Model':<30} {'Val RMSE':<12} {'Train RMSE':<12} {'Time (min)':<12}")
        print("-"*100)

        for rank, r in enumerate(successful, 1):
            print(f"{rank:<6} {r['name']:<30} {r['final_val_rmse']:<12.6f} "
                  f"{r['final_train_rmse']:<12.6f} {r['training_time_seconds']/60:<12.2f}")

        print("-"*100)
        print(f"\n🏆 BEST MODEL: {successful[0]['name']}")
        print(f"   Validation RMSE: {successful[0]['final_val_rmse']:.6f}")
        print(f"   Description: {successful[0]['description']}")

    if failed:
        print(f"\n\nFailed Models ({len(failed)}):")
        print("-"*100)
        for r in failed:
            print(f"✗ {r['name']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*100)


def save_results(results, overall_time):
    """Save results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, f'experiment_results_{timestamp}.json')

    output_data = {
        'timestamp': timestamp,
        'configuration': {
            'num_epochs': NUMEPOCHS,
            'thread_count': THREAD_COUNT,
            'data_percentage': GLOBAL_PERCENTAGE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        },
        'overall_time_seconds': overall_time,
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Also save a CSV for easy viewing
    csv_file = os.path.join(results_dir, f'experiment_results_{timestamp}.csv')
    try:
        import csv
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Description', 'Status', 'Val RMSE', 'Train RMSE', 'Time (min)'])
            for r in results:
                if r['status'] == 'success':
                    writer.writerow([
                        r['name'],
                        r['description'],
                        r['status'],
                        f"{r['final_val_rmse']:.6f}",
                        f"{r['final_train_rmse']:.6f}",
                        f"{r['training_time_seconds']/60:.2f}"
                    ])
                else:
                    writer.writerow([
                        r['name'],
                        r['description'],
                        r['status'],
                        'N/A',
                        'N/A',
                        f"{r['training_time_seconds']/60:.2f}"
                    ])
        logger.info(f"CSV results saved to: {csv_file}")
    except Exception as e:
        logger.warning(f"Could not save CSV: {e}")


def main():
    """Main entry point"""
    if THREAD_COUNT <= 1:
        return run_experiments_sequential()
    else:
        return run_experiments_parallel()


if __name__ == "__main__":
    results = main()
