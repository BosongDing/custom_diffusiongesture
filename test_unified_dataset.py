#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import time

# Add the scripts directory to Python path
sys.path.append('scripts')

# Import our unified dataloader
from scripts.data_loader.unified_data_loader import (
    UnifiedMotionDataset, 
    unified_collate_fn, 
    create_unified_dataset,
    DatasetComparator
)

# Import individual dataloaders for comparison
from scripts.data_loader.lmdb_data_loader_new import LMDBDataset
try:
    from scripts.data_loader.lmdb_data_loader_expressive import SpeechMotionDataset
except:
    print("Warning: Could not import SpeechMotionDataset - TED Expressive tests will be skipped")
    SpeechMotionDataset = None

def test_individual_datasets(trinity_path=None, beat_path=None, ted_expressive_path=None):
    """Test loading individual datasets separately"""
    print("="*60)
    print("TESTING INDIVIDUAL DATASETS")
    print("="*60)
    
    individual_datasets = {}
    
    # Test Trinity dataset
    if trinity_path and os.path.exists(trinity_path):
        print(f"\nTesting Trinity dataset from: {trinity_path}")
        try:
            trinity_dataset = LMDBDataset(dataset_path=trinity_path)
            individual_datasets['trinity'] = trinity_dataset
            
            print(f"  ✓ Successfully loaded {len(trinity_dataset)} samples")
            
            # Test a few samples
            for i in range(min(3, len(trinity_dataset))):
                sample = trinity_dataset[i]
                word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
                print("TRINITY",type(pose_seq), type(vec_seq), type(audio), type(spectrogram), type(aux_info))
                print(f"  Sample {i}: pose_seq shape {pose_seq.shape}, vec_seq shape {vec_seq.shape}")
                
        except Exception as e:
            print(f"  ✗ Failed to load Trinity dataset: {e}")
    
    # Test BEAT dataset  
    if beat_path and os.path.exists(beat_path):
        print(f"\nTesting BEAT dataset from: {beat_path}")
        try:
            beat_dataset = LMDBDataset(dataset_path=beat_path)
            individual_datasets['beat'] = beat_dataset
            
            print(f"  ✓ Successfully loaded {len(beat_dataset)} samples")
            
            # Test a few samples
            for i in range(min(3, len(beat_dataset))):
                sample = beat_dataset[i]
                word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
                print("BEAT",type(pose_seq), type(vec_seq), type(audio), type(spectrogram), type(aux_info))
                print(f"  Sample {i}: pose_seq shape {pose_seq.shape}, vec_seq shape {vec_seq.shape}")
                
        except Exception as e:
            print(f"  ✗ Failed to load BEAT dataset: {e}")
    
    # Test TED Expressive dataset
    if ted_expressive_path and os.path.exists(ted_expressive_path) and SpeechMotionDataset:
        print(f"\nTesting TED Expressive dataset from: {ted_expressive_path}")
        #load form yml
        import yaml
        with open('config/pose_diffusion_expressive.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        mean_pose = np.array(config['mean_pose'])
        mean_dir_vec = np.array(config['mean_dir_vec'])
        # mean_pose = np.zeros(126 // 3 * 3)  # Default for TED Expressive
        # mean_dir_vec = np.zeros(126)
        
        ted_dataset = SpeechMotionDataset(
            lmdb_dir=ted_expressive_path,
            n_poses=34,
            subdivision_stride=10,
            pose_resampling_fps=15,
            mean_pose=mean_pose,
            mean_dir_vec=mean_dir_vec,
            speaker_model=0  # Disable speaker model for testing
        )
        individual_datasets['ted_expressive'] = ted_dataset
        
        print(f"  ✓ Successfully loaded {len(ted_dataset)} samples")
        
        # Test a few samples with better error handling
        for i in range(min(3, len(ted_dataset))):
            sample = ted_dataset[i]
            word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
            print("TED",type(pose_seq), type(vec_seq), type(audio), type(spectrogram), type(aux_info))
        # try:
        #     # Load mean vectors if available
        #     mean_pose = np.zeros(126 // 3 * 3)  # Default for TED Expressive
        #     mean_dir_vec = np.zeros(126)
            
        #     ted_dataset = SpeechMotionDataset(
        #         lmdb_dir=ted_expressive_path,
        #         n_poses=34,
        #         subdivision_stride=10,
        #         pose_resampling_fps=15,
        #         mean_pose=mean_pose,
        #         mean_dir_vec=mean_dir_vec,
        #         speaker_model=0  # Disable speaker model for testing
        #     )
            
        #     # Try to set a dummy language model to avoid word processing errors
        #     try:
        #         from scripts.model.vocab import Vocab
        #         dummy_lang_model = Vocab('dummy')
        #         dummy_lang_model.n_words = 4  # SOS, EOS, UNK, PAD
        #         ted_dataset.set_lang_model(dummy_lang_model)
        #     except:
        #         pass  # If vocab import fails, continue without language model
            
        #     individual_datasets['ted_expressive'] = ted_dataset
            
        #     print(f"  ✓ Successfully loaded {len(ted_dataset)} samples")
            
        #     # Test a few samples with better error handling
        #     for i in range(min(3, len(ted_dataset))):
        #         sample = ted_dataset[i]
        #         word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
        #         print("TED",type(pose_seq), type(vec_seq), type(audio), type(spectrogram), type(aux_info))
        #         # try:
                #     sample = ted_dataset[i]
                #     word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
                #     print("TED",type(pose_seq), type(vec_seq), type(audio), type(spectrogram), type(aux_info))
                #     # Check types and shapes
                #     pose_type = type(pose_seq)
                #     vec_type = type(vec_seq)
                    
                #     if hasattr(pose_seq, 'shape'):
                #         pose_shape = pose_seq.shape
                #     else:
                #         pose_shape = f"list of length {len(pose_seq) if hasattr(pose_seq, '__len__') else 'unknown'}"
                    
                #     if hasattr(vec_seq, 'shape'):
                #         vec_shape = vec_seq.shape
                #     else:
                #         vec_shape = f"list of length {len(vec_seq) if hasattr(vec_seq, '__len__') else 'unknown'}"
                    
                #     print(f"  Sample {i}: pose_seq {pose_type} {pose_shape}, vec_seq {vec_type} {vec_shape}")
                    
                # except Exception as sample_error:
                #     print(f"  Sample {i}: ✗ Error loading sample: {sample_error}")
                #     break
                
        # except Exception as e:
        #     print(f"  ✗ Failed to load TED Expressive dataset: {e}")
        #     import traceback
        #     print(f"  Full traceback: {traceback.format_exc()}")
    
    return individual_datasets

def test_unified_dataset(trinity_path=None, beat_path=None, ted_expressive_path=None):
    """Test the unified dataset"""
    print("\n" + "="*60)
    print("TESTING UNIFIED DATASET")
    print("="*60)
    
    try:
        # Create unified dataset
        unified_dataset = create_unified_dataset(
            trinity_path=trinity_path,
            beat_path=beat_path,
            ted_expressive_path=ted_expressive_path
        )
        
        print(f"\n✓ Successfully created unified dataset with {len(unified_dataset)} total samples")
        
        # Print dataset info
        dataset_info = unified_dataset.get_dataset_info()
        print("\nDataset Information:")
        for name, info in dataset_info['datasets'].items():
            print(f"  {name}: {info['count']} samples, dim: {info['pose_dim']}")
        
        # Test sampling from each dataset
        print("\nTesting samples from each dataset:")
        for dataset_name in dataset_info['datasets'].keys():
            try:
                sample_indices = unified_dataset.get_samples_by_dataset(dataset_name)
                if sample_indices:
                    sample = unified_dataset[sample_indices[0]]
                    word_seq, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info, dataset_info_sample = sample
                    
                    # Check data types and shapes
                    pose_type = type(pose_seq)
                    pose_shape = pose_seq.shape if hasattr(pose_seq, 'shape') else 'no shape attr'
                    
                    print(f"  {dataset_name}: pose {pose_type} {pose_shape}, dataset_info: {dataset_info_sample}")
            except Exception as sample_error:
                print(f"  {dataset_name}: ✗ Error testing sample: {sample_error}")
        
        return unified_dataset
        
    except Exception as e:
        print(f"✗ Failed to create unified dataset: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
        # Try creating with just working datasets
        print("\nTrying to create unified dataset with only working datasets...")
        try:
            # Try Trinity + BEAT only
            if trinity_path and beat_path:
                unified_dataset = create_unified_dataset(
                    trinity_path=trinity_path,
                    beat_path=beat_path,
                    ted_expressive_path=None
                )
                print(f"✓ Created unified dataset with Trinity + BEAT: {len(unified_dataset)} samples")
                return unified_dataset
            elif trinity_path:
                unified_dataset = create_unified_dataset(
                    trinity_path=trinity_path,
                    beat_path=None,
                    ted_expressive_path=None
                )
                print(f"✓ Created unified dataset with Trinity only: {len(unified_dataset)} samples")
                return unified_dataset
            elif beat_path:
                unified_dataset = create_unified_dataset(
                    trinity_path=None,
                    beat_path=beat_path,
                    ted_expressive_path=None
                )
                print(f"✓ Created unified dataset with BEAT only: {len(unified_dataset)} samples")
                return unified_dataset
        except Exception as fallback_error:
            print(f"✗ Fallback also failed: {fallback_error}")
        
        return None

def test_data_consistency(unified_dataset):
    """Test that unified dataset returns same data as individual datasets"""
    print("\n" + "="*60)
    print("TESTING DATA CONSISTENCY")
    print("="*60)
    
    try:
        comparator = DatasetComparator(unified_dataset)
        results = comparator.compare_samples(num_samples=10)
        
        print("\nConsistency Test Results:")
        all_passed = True
        for dataset_name, result in results.items():
            matches = result['matches']
            total = result['total_compared']
            success_rate = (matches / total * 100) if total > 0 else 0
            
            print(f"  {dataset_name}: {matches}/{total} matches ({success_rate:.1f}%)")
            if success_rate < 100:
                all_passed = False
        
        if all_passed:
            print("\n✓ All consistency tests passed!")
        else:
            print("\n✗ Some consistency tests failed!")
            
        return all_passed
        
    except Exception as e:
        print(f"✗ Consistency test failed: {e}")
        return False

def test_dataloader_and_collate(unified_dataset):
    """Test the DataLoader with unified collate function"""
    print("\n" + "="*60)
    print("TESTING DATALOADER AND COLLATE FUNCTION")
    print("="*60)
    
    try:
        # Create dataloader
        dataloader = DataLoader(
            unified_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=unified_collate_fn,
            num_workers=0  # Use 0 for debugging
        )
        
        print(f"✓ Created DataLoader with batch_size=8")
        
        # Test a few batches
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Contains datasets: {list(batch.keys())}")
            
            for dataset_name, dataset_batch in batch.items():
                batch_size = dataset_batch['batch_size']
                pose_shape = dataset_batch['pose_seqs'].shape
                vec_shape = dataset_batch['vec_seqs'].shape
                audio_shape = dataset_batch['audios'].shape
                
                print(f"  {dataset_name}: batch_size={batch_size}")
                print(f"    pose_seqs: {pose_shape}")
                print(f"    vec_seqs: {vec_shape}")
                print(f"    audios: {audio_shape}")
            
            if batch_idx >= 2:  # Test only first 3 batches
                break
        
        print("\n✓ DataLoader and collate function working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        return False

def test_filtering(unified_dataset):
    """Test dataset filtering functionality"""
    print("\n" + "="*60)
    print("TESTING DATASET FILTERING")
    print("="*60)
    
    try:
        dataset_info = unified_dataset.get_dataset_info()
        available_datasets = list(dataset_info['datasets'].keys())
        
        print(f"Available datasets: {available_datasets}")
        
        # Test filtering by single dataset
        for dataset_name in available_datasets:
            filtered_dataset = unified_dataset.filter_by_dataset(dataset_name)
            expected_count = dataset_info['datasets'][dataset_name]['count']
            actual_count = len(filtered_dataset)
            
            print(f"  Filter by '{dataset_name}': {actual_count} samples (expected: {expected_count})")
            
            if actual_count != expected_count:
                print(f"    ✗ Count mismatch!")
                return False
            
            # Test a sample from filtered dataset
            if actual_count > 0:
                sample = filtered_dataset[0]
                dataset_info_sample = sample[-1]  # dataset_info is last element
                
                if dataset_info_sample['name'] != dataset_name:
                    print(f"    ✗ Wrong dataset in filtered view!")
                    return False
        
        # Test filtering by multiple datasets if we have more than one
        if len(available_datasets) > 1:
            multi_filter = unified_dataset.filter_by_dataset(available_datasets[:2])
            expected_total = sum(dataset_info['datasets'][name]['count'] for name in available_datasets[:2])
            actual_total = len(multi_filter)
            
            print(f"  Multi-filter by {available_datasets[:2]}: {actual_total} samples (expected: {expected_total})")
            
            if actual_total != expected_total:
                print(f"    ✗ Multi-filter count mismatch!")
                return False
        
        print("\n✓ All filtering tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Filtering test failed: {e}")
        return False

def test_performance(unified_dataset):
    """Test loading performance"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE")
    print("="*60)
    
    try:
        num_samples = min(100, len(unified_dataset))
        
        # Test random access performance
        print(f"Testing random access to {num_samples} samples...")
        start_time = time.time()
        
        for i in range(num_samples):
            sample = unified_dataset[i]
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_samples * 1000  # ms per sample
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per sample: {avg_time:.3f}ms")
        
        # Test dataloader performance
        print(f"\nTesting DataLoader performance...")
        dataloader = DataLoader(
            unified_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=unified_collate_fn,
            num_workers=0
        )
        
        start_time = time.time()
        samples_processed = 0
        
        for batch_idx, batch in enumerate(dataloader):
            for dataset_name, dataset_batch in batch.items():
                samples_processed += dataset_batch['batch_size']
            
            if batch_idx >= 10:  # Test first 10 batches
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = samples_processed / total_time
        
        print(f"  Processed {samples_processed} samples in {total_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} samples/second")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test unified dataset functionality')
    parser.add_argument('--trinity_path', type=str, 
                       default="./data/trinity_all_cache",
                       help='Path to Trinity dataset cache')
    parser.add_argument('--beat_path', type=str,
                       default="./data/beat_english_v0.2.1/beat_all_cache", 
                       help='Path to BEAT dataset cache')
    parser.add_argument('--ted_expressive_path', type=str,
                       default="./data/ted_expressive_dataset/train",
                       help='Path to TED Expressive dataset')
    parser.add_argument('--skip_individual', action='store_true',
                       help='Skip testing individual datasets')
    parser.add_argument('--skip_consistency', action='store_true',
                       help='Skip consistency tests')
    parser.add_argument('--skip_performance', action='store_true',
                       help='Skip performance tests')
    
    args = parser.parse_args()
    
    print("UNIFIED DATASET TEST SUITE")
    print("="*60)
    print(f"Trinity path: {args.trinity_path}")
    print(f"BEAT path: {args.beat_path}")
    print(f"TED Expressive path: {args.ted_expressive_path}")
    
    # Check which datasets are available
    available_datasets = []
    if args.trinity_path and os.path.exists(args.trinity_path):
        available_datasets.append('Trinity')
    if args.beat_path and os.path.exists(args.beat_path):
        available_datasets.append('BEAT')
    if args.ted_expressive_path and os.path.exists(args.ted_expressive_path):
        available_datasets.append('TED Expressive')
    
    print(f"Available datasets: {', '.join(available_datasets)}")
    
    if not available_datasets:
        print("ERROR: No datasets found! Please check your paths.")
        return
    
    # Test individual datasets
    individual_datasets = {}
    if not args.skip_individual:
        individual_datasets = test_individual_datasets(
            trinity_path=args.trinity_path if os.path.exists(args.trinity_path or '') else None,
            beat_path=args.beat_path if os.path.exists(args.beat_path or '') else None,
            ted_expressive_path=args.ted_expressive_path if os.path.exists(args.ted_expressive_path or '') else None
        )
    
    # Test unified dataset
    unified_dataset = test_unified_dataset(
        trinity_path=args.trinity_path if os.path.exists(args.trinity_path or '') else None,
        beat_path=args.beat_path if os.path.exists(args.beat_path or '') else None,
        ted_expressive_path=args.ted_expressive_path if os.path.exists(args.ted_expressive_path or '') else None
    )
    
    if unified_dataset is None:
        print("ERROR: Could not create unified dataset!")
        return
    
    # Run tests
    test_results = []
    
    # Test data consistency
    if not args.skip_consistency:
        test_results.append(('Consistency', test_data_consistency(unified_dataset)))
    
    # Test dataloader and collate function
    test_results.append(('DataLoader', test_dataloader_and_collate(unified_dataset)))
    
    # Test filtering
    test_results.append(('Filtering', test_filtering(unified_dataset)))
    
    # Test performance
    if not args.skip_performance:
        test_results.append(('Performance', test_performance(unified_dataset)))
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:15}: {status}")
        if not result:
            all_passed = False
    
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

if __name__ == "__main__":
    main() 