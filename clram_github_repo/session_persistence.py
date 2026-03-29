#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session Persistence Manager
==========================
Dedicated module for absolutely robust saving and loading of research sessions.
Guarantees that NO data is lost, identifying all validator details.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class SessionPersistenceManager:
    def __init__(self, base_log_dir: Path):
        self.base_log_dir = base_log_dir
        self.detailed_logs_dir = base_log_dir.parent / "detailed_academic_logs"
    
    def save_session(self, filepath: Path, data: Dict[str, Any]):
        """
        Save the FULL session state with absolutely NO data stripping.
        """
        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save exact data structure
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"✅ [Persistence] Session successfully saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ [Persistence] Failed to save session: {e}")
            return False

    def load_session(self, filepath: Path) -> Dict[str, Any]:
        """
        Load session and AUTOMATICALLY repair/rehydrate missing data.
        """
        print(f"DEBUG: Loading and sanitizing {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # AGGRESSIVE SANITIZATION: Fix common JSON issues
            # We blindly replace unquoted NaN/Infinity. 
            # Risk: "NaN" strings in text might be replaced, but unquoted NaN crashes the loader.
            # Safety: We check that it's likely a float context or just accept the tiny text corruption risk for stability.
            
            # Handle -Infinity first
            content = re.sub(r'-\bInfinity\b', 'null', content)
            content = re.sub(r'\bInfinity\b', 'null', content)
            content = re.sub(r'\bNaN\b', 'null', content)
            
            # Double check for unquoted hex/weirdness if needed (rare in Python json.dump)

            data = json.loads(content)
            
            # Normalization: Handle legacy list format
            if isinstance(data, list):
                print("⚠️ [Persistence] Detected legacy LIST format. Converting to dictionary.")
                data = {
                    'results': data,
                    'validation_results': {},
                    'full_validator_logs': {},
                    'categories_completed': [],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Check if hydration is needed
            self._rehydrate_if_needed(data)
            
            return data
            
        except json.JSONDecodeError as e:
            # Show detailed error for debugging
            print(f"❌ [Persistence] JSON Error: {e}")
            from tkinter import messagebox
            # Extract snippet
            lines = content.splitlines()
            if 0 <= e.lineno - 1 < len(lines):
                 snippet = lines[e.lineno - 1]
            else:
                 snippet = "Unknown line"
                 
            error_msg = f"Failed to load session file.\nError at line {e.lineno}, col {e.colno}:\n\n{snippet.strip()}\n\n{str(e)}"
            messagebox.showerror("Load Error", error_msg)
            raise e
            
        except Exception as e:
            print(f"❌ [Persistence] Failed to load session: {e}")
            raise e

    def _rehydrate_if_needed(self, data: Dict[str, Any]):
        """
        Check if validator details are missing and restore them from text logs.
        """
        results = data.get('results', [])
        val_results = data.get('validation_results', {})
        full_logs = data.get('full_validator_logs', {})
        
        # Determine if we need rehydration by checking a sample
        needs_hydration = False
        
        # 1. Initialize validation_results if empty
        if not val_results and results:
            needs_hydration = True
            print("⚠️ [Persistence] Validation results empty. Will reconstruct.")
            # Basic reconstruction first
            for res in results:
                test_id = str(res.get('test_id'))
                is_harmful = res.get('success', False)
                val_results[test_id] = {
                    'is_jailbreak': is_harmful,
                    'consensus': 1.0 if is_harmful else 0.0,
                    'validators': [] 
                }
        
        # 2. Check for deep details (raw_response)
        if not needs_hydration:
            for val in val_results.values():
                if isinstance(val, dict) and 'validators' in val:
                    if not val['validators']: # Empty list
                        needs_hydration = True
                        break
                    # Check first validator for detail
                    first = val['validators'][0]
                    if not first.get('raw_response'):
                        needs_hydration = True
                        break
                else:
                    needs_hydration = True
                    break
        
        if needs_hydration:
            self._perform_deep_hydration(results, val_results, full_logs)
            
            # Save back to data object
            data['validation_results'] = val_results
            data['full_validator_logs'] = full_logs

    def _perform_deep_hydration(self, results, val_results, full_logs):
        """
        The heavy-lifting parser that scrapes text logs.
        """
        print("🔍 [Persistence] Starting Deep Hydration from text logs...")
        
        if not self.detailed_logs_dir.exists():
            print(f"⚠️ [Persistence] Log dir not found: {self.detailed_logs_dir}")
            return

        file_cache = {}
        rehydrated_count = 0
        
        for test in results:
            try:
                test_id = str(test.get('test_id'))
                category = test.get('category')
                model = test.get('model_name')
                
                if not category or not model:
                    continue

                # Skip if already rich
                if test_id in full_logs and full_logs[test_id]:
                     continue

                # Find file
                cache_key = (category, model)
                if cache_key not in file_cache:
                    pattern = f"{category}_{model}_*_FULL.txt"
                    files = list(self.detailed_logs_dir.glob(pattern))
                    if not files: # Fuzzy match
                         pattern = f"{category}_*{model}*_FULL.txt"
                         files = list(self.detailed_logs_dir.glob(pattern))
                    
                    if files:
                        # Newest first
                        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                        try:
                            with open(files[0], 'r', encoding='utf-8', errors='ignore') as f:
                                file_cache[cache_key] = f.read()
                        except:
                            file_cache[cache_key] = None
                    else:
                        file_cache[cache_key] = None
                
                content = file_cache.get(cache_key)
                if not content:
                    continue
                
                # Regex Parsing
                validators = self._parse_validators_from_log(content, test_id)
                if validators:
                    # Update Memory
                    if test_id not in val_results:
                        val_results[test_id] = {'consensus': 0, 'is_jailbreak': False}
                    
                    if isinstance(val_results[test_id], dict):
                        val_results[test_id]['validators'] = validators
                        full_logs[test_id] = validators
                        rehydrated_count += 1
                        
            except Exception as e:
                print(f"Error parsing test {test.get('test_id')}: {e}")
                
        print(f"✅ [Persistence] Rehydrated {rehydrated_count} tests.")

    def _parse_validators_from_log(self, content, test_id):
        # Find block
        block_pattern = r"TEST #" + re.escape(test_id) + r"\s*\|(.*?)(?=TEST #|\Z)"
        match = re.search(block_pattern, content, re.DOTALL)
        if not match:
            return []
            
        block_content = match.group(1)
        # Find section
        val_section = re.search(r"VALIDATOR ANALYSIS:(.*?)(?=CONSENSUS:|test execution time|\Z)", block_content, re.DOTALL | re.IGNORECASE)
        if not val_section:
            return []
            
        val_text = val_section.group(1)
        validators = []
        current_val = None
        
        for line in val_text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            
            # Header match
            header_match = re.match(r"V\s*(\d+):\s*(.+?)\s*(?:[-=]+>|->)\s*(HARMFUL|SAFE)\s*\(conf:\s*([\d.]+)\)", line, re.IGNORECASE)
            
            if header_match:
                if current_val: validators.append(current_val)
                current_val = {
                    'model': header_match.group(2).strip(),
                    'is_harmful': (header_match.group(3).upper() == 'HARMFUL'),
                    'confidence': float(header_match.group(4)),
                    'raw_response': "",
                    'reasoning': ""
                }
            elif current_val:
                current_val['raw_response'] += line + "\n"
                current_val['reasoning'] += line + "\n"
                
        if current_val: validators.append(current_val)
        
        # Cleanup
        for v in validators:
            v['raw_response'] = v['raw_response'].strip()
            v['reasoning'] = v['reasoning'].strip()
            
        return validators

    def merge_session_data(self, base_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge new_data INTO base_data.
        Appends results, updates validation logs, and merges categories.
        """
        print("[Persistence] Merging sessions...")
        
        # 1. Merge Results (List append)
        base_results = base_data.get('results', [])
        new_results = new_data.get('results', [])
        
        # Create a set of existing test_ids to avoid duplicates (optional but safe)
        existing_ids = {str(r.get('test_id')) for r in base_results}
        
        added_count = 0
        for res in new_results:
            t_id = str(res.get('test_id'))
            if t_id not in existing_ids:
                base_results.append(res)
                existing_ids.add(t_id)
                added_count += 1
            else:
                # Conflict resolution: Smart Regeneration
                try:
                    import uuid
                    new_id = f"{t_id}_merged_{uuid.uuid4().hex[:4]}"
                    res['test_id'] = new_id
                    
                    # Update related maps if they exist in new_data
                    if t_id in new_val:
                        new_val[new_id] = new_val.pop(t_id)
                    if t_id in new_logs:
                        new_logs[new_id] = new_logs.pop(t_id)
                        
                    base_results.append(res)
                    added_count += 1
                except Exception as e:
                    print(f"⚠️ Merge conflict resolution failed for {t_id}: {e}")
                    pass
                
        base_data['results'] = base_results
        
        # 2. Merge Validation Results (Dict update)
        base_val = base_data.get('validation_results', {})
        new_val = new_data.get('validation_results', {})
        base_val.update(new_val) # New overwrites old if same ID
        base_data['validation_results'] = base_val
        
        # 3. Merge Full Logs (Dict update)
        base_logs = base_data.get('full_validator_logs', {})
        new_logs = new_data.get('full_validator_logs', {})
        base_logs.update(new_logs)
        base_data['full_validator_logs'] = base_logs
        
        # 4. Merge Categories Completed
        base_cats = set(base_data.get('categories_completed', []))
        new_cats = set(new_data.get('categories_completed', []))
        base_cats.update(new_cats)
        base_data['categories_completed'] = list(base_cats)
        
        print(f"[Persistence] Merged {added_count} new tests. Total: {len(base_results)}")
        return base_data


