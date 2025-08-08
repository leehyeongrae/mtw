"""
Shared Memory management using mmap for multiprocessing data sharing
"""
import mmap
import struct
import json
import os
import time
import tempfile
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("shared_memory")

class MemoryMap:
    """Memory mapped file manager"""
    
    def __init__(self, name: str, size: int = 1024 * 1024 * 50):  # 50MB default
        self.name = name
        self.size = size
        self.file_path = Path(tempfile.gettempdir()) / f"mtw_{name}.mmap"
        self.file_handle = None
        self.mmap_obj = None
        self._init_mmap()
    
    def _init_mmap(self):
        """Initialize memory mapped file"""
        try:
            # Create or open file
            self.file_handle = open(self.file_path, "r+b" if self.file_path.exists() else "w+b")
            
            if not self.file_path.exists() or os.path.getsize(self.file_path) < self.size:
                # Initialize with zeros
                self.file_handle.write(b'\x00' * self.size)
                self.file_handle.flush()
            
            # Create memory map
            self.mmap_obj = mmap.mmap(self.file_handle.fileno(), self.size)
            logger.info(f"Initialized memory map: {self.name} ({self.size} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory map {self.name}: {e}")
            raise
    
    def read_bytes(self, offset: int, length: int) -> bytes:
        """Read bytes from memory map"""
        try:
            if offset + length > self.size:
                raise ValueError(f"Read beyond buffer: {offset + length} > {self.size}")
            return self.mmap_obj[offset:offset + length]
        except Exception as e:
            logger.error(f"Error reading from {self.name}: {e}")
            return b''
    
    def write_bytes(self, offset: int, data: bytes) -> bool:
        """Write bytes to memory map"""
        try:
            if offset + len(data) > self.size:
                logger.warning(f"Write beyond buffer: {offset + len(data)} > {self.size}")
                return False
            
            self.mmap_obj[offset:offset + len(data)] = data
            self.mmap_obj.flush()
            return True
        except Exception as e:
            logger.error(f"Error writing to {self.name}: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.mmap_obj:
                self.mmap_obj.close()
            if self.file_handle:
                self.file_handle.close()
            if self.file_path.exists():
                os.unlink(self.file_path)
            logger.info(f"Cleaned up memory map: {self.name}")
        except Exception as e:
            logger.error(f"Error cleaning up {self.name}: {e}")

class SharedDataStructure:
    """Base class for shared data structures"""
    
    def __init__(self, name: str):
        self.name = name
        self.mmap = MemoryMap(name)
        self.header_size = 1024  # Reserve 1KB for headers
        self.data_offset = self.header_size
    
    def _pack_header(self, **kwargs) -> bytes:
        """Pack header information"""
        header_info = {
            'timestamp': time.time(),
            'version': 1,
            **kwargs
        }
        header_json = json.dumps(header_info).encode('utf-8')
        # Pad to header size
        return header_json.ljust(self.header_size, b'\x00')
    
    def _unpack_header(self) -> Dict:
        """Unpack header information"""
        try:
            header_bytes = self.mmap.read_bytes(0, self.header_size)
            header_json = header_bytes.rstrip(b'\x00').decode('utf-8')
            return json.loads(header_json) if header_json else {}
        except Exception as e:
            logger.debug(f"Error unpacking header for {self.name}: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources"""
        self.mmap.cleanup()

class SharedCandles(SharedDataStructure):
    """Shared candle data structure"""
    
    def __init__(self, symbol: str):
        super().__init__(f"candles_{symbol}")
        self.symbol = symbol
    
    def write_candles(self, historical_df: Optional[pd.DataFrame] = None, 
                     current_candle: Optional[Dict] = None) -> bool:
        """Write candle data to shared memory"""
        try:
            data_to_store = {}
            
            if historical_df is not None:
                # Convert DataFrame to binary format
                data_to_store['historical'] = {
                    'data': historical_df.to_dict('records'),
                    'length': len(historical_df),
                    'columns': list(historical_df.columns)
                }
            
            if current_candle is not None:
                data_to_store['current'] = current_candle
            
            # Serialize data
            serialized = json.dumps(data_to_store).encode('utf-8')
            
            # Write header
            header = self._pack_header(
                data_length=len(serialized),
                has_historical=historical_df is not None,
                has_current=current_candle is not None
            )
            
            # Check if data fits
            if len(serialized) + self.header_size > self.mmap.size:
                logger.warning(f"Data too large for {self.symbol}: {len(serialized)} bytes")
                return False
            
            # Write to memory map
            success = self.mmap.write_bytes(0, header)
            if success:
                success = self.mmap.write_bytes(self.data_offset, serialized)
            
            return success
            
        except Exception as e:
            logger.error(f"Error writing candles for {self.symbol}: {e}")
            return False
    
    def read_candles(self) -> tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Read candle data from shared memory"""
        try:
            header = self._unpack_header()
            if not header:
                return None, None
            
            data_length = header.get('data_length', 0)
            if data_length == 0:
                return None, None
            
            # Read serialized data
            serialized_bytes = self.mmap.read_bytes(self.data_offset, data_length)
            data = json.loads(serialized_bytes.decode('utf-8'))
            
            historical_df = None
            current_candle = None
            
            # Reconstruct historical DataFrame
            if 'historical' in data:
                hist_data = data['historical']
                if hist_data['data']:
                    historical_df = pd.DataFrame(hist_data['data'])
                    # Ensure column order
                    if hist_data['columns']:
                        historical_df = historical_df[hist_data['columns']]
            
            # Get current candle
            if 'current' in data:
                current_candle = data['current']
            
            return historical_df, current_candle
            
        except Exception as e:
            logger.error(f"Error reading candles for {self.symbol}: {e}")
            return None, None

class SharedIndicators(SharedDataStructure):
    """Shared indicators data structure"""
    
    def __init__(self, symbol: str):
        super().__init__(f"indicators_{symbol}")
        self.symbol = symbol
    
    def write_indicators(self, indicators: Dict) -> bool:
        """Write indicators to shared memory"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            safe_indicators = {}
            for key, value in indicators.items():
                if isinstance(value, np.ndarray):
                    safe_indicators[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    safe_indicators[key] = list(value)
                else:
                    safe_indicators[key] = value
            
            # Serialize data
            serialized = json.dumps(safe_indicators).encode('utf-8')
            
            # Write header
            header = self._pack_header(
                data_length=len(serialized),
                indicator_count=len(safe_indicators)
            )
            
            # Check if data fits
            if len(serialized) + self.header_size > self.mmap.size:
                logger.warning(f"Indicators too large for {self.symbol}: {len(serialized)} bytes")
                return False
            
            # Write to memory map
            success = self.mmap.write_bytes(0, header)
            if success:
                success = self.mmap.write_bytes(self.data_offset, serialized)
            
            return success
            
        except Exception as e:
            logger.error(f"Error writing indicators for {self.symbol}: {e}")
            return False
    
    def read_indicators(self) -> Optional[Dict]:
        """Read indicators from shared memory"""
        try:
            header = self._unpack_header()
            if not header:
                return None
            
            data_length = header.get('data_length', 0)
            if data_length == 0:
                return None
            
            # Read serialized data
            serialized_bytes = self.mmap.read_bytes(self.data_offset, data_length)
            indicators = json.loads(serialized_bytes.decode('utf-8'))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error reading indicators for {self.symbol}: {e}")
            return None

class SharedSignals(SharedDataStructure):
    """Shared signals data structure"""
    
    def __init__(self, symbol: str):
        super().__init__(f"signals_{symbol}")
        self.symbol = symbol
    
    def write_signal(self, signal: Dict) -> bool:
        """Write signal to shared memory"""
        try:
            # Add timestamp
            signal_data = dict(signal)
            signal_data['timestamp'] = time.time()
            signal_data['symbol'] = self.symbol
            
            # Serialize data
            serialized = json.dumps(signal_data).encode('utf-8')
            
            # Write header
            header = self._pack_header(
                data_length=len(serialized),
                signal_action=signal.get('action', '')
            )
            
            # Check if data fits
            if len(serialized) + self.header_size > self.mmap.size:
                logger.warning(f"Signal too large for {self.symbol}: {len(serialized)} bytes")
                return False
            
            # Write to memory map
            success = self.mmap.write_bytes(0, header)
            if success:
                success = self.mmap.write_bytes(self.data_offset, serialized)
            
            return success
            
        except Exception as e:
            logger.error(f"Error writing signal for {self.symbol}: {e}")
            return False
    
    def read_signal(self) -> Optional[Dict]:
        """Read signal from shared memory"""
        try:
            header = self._unpack_header()
            if not header:
                return None
            
            data_length = header.get('data_length', 0)
            if data_length == 0:
                return None
            
            # Read serialized data
            serialized_bytes = self.mmap.read_bytes(self.data_offset, data_length)
            signal = json.loads(serialized_bytes.decode('utf-8'))
            
            return signal
            
        except Exception as e:
            logger.error(f"Error reading signal for {self.symbol}: {e}")
            return None

class SharedMetadata(SharedDataStructure):
    """Shared metadata (symbol list, positions, etc.)"""
    
    def __init__(self):
        super().__init__("metadata")
    
    def write_symbol_list(self, symbols: List[str]) -> bool:
        """Write symbol list to shared memory"""
        return self._write_data('symbols', symbols)
    
    def read_symbol_list(self) -> List[str]:
        """Read symbol list from shared memory"""
        return self._read_data('symbols', [])
    
    def write_positions(self, positions: Dict) -> bool:
        """Write positions to shared memory"""
        return self._write_data('positions', positions)
    
    def read_positions(self) -> Dict:
        """Read positions from shared memory"""
        return self._read_data('positions', {})
    
    def write_system_status(self, status: Dict) -> bool:
        """Write system status to shared memory"""
        return self._write_data('system_status', status)
    
    def read_system_status(self) -> Dict:
        """Read system status from shared memory"""
        return self._read_data('system_status', {})
    
    def _write_data(self, key: str, data: Any) -> bool:
        """Generic write method"""
        try:
            # Serialize data
            serialized = json.dumps({key: data}).encode('utf-8')
            
            # Write header
            header = self._pack_header(
                data_length=len(serialized),
                data_key=key
            )
            
            # Check if data fits
            if len(serialized) + self.header_size > self.mmap.size:
                logger.warning(f"Data too large for {key}: {len(serialized)} bytes")
                return False
            
            # Write to memory map
            success = self.mmap.write_bytes(0, header)
            if success:
                success = self.mmap.write_bytes(self.data_offset, serialized)
            
            return success
            
        except Exception as e:
            logger.error(f"Error writing {key}: {e}")
            return False
    
    def _read_data(self, key: str, default: Any = None) -> Any:
        """Generic read method"""
        try:
            header = self._unpack_header()
            if not header:
                return default
            
            data_length = header.get('data_length', 0)
            if data_length == 0:
                return default
            
            # Read serialized data
            serialized_bytes = self.mmap.read_bytes(self.data_offset, data_length)
            data_dict = json.loads(serialized_bytes.decode('utf-8'))
            
            return data_dict.get(key, default)
            
        except Exception as e:
            logger.error(f"Error reading {key}: {e}")
            return default

class SharedMemoryManager:
    """Centralized manager for all shared memory structures"""
    
    def __init__(self):
        self.logger = get_logger("shared_memory_manager")
        self.candles = {}  # symbol -> SharedCandles
        self.indicators = {}  # symbol -> SharedIndicators
        self.signals = {}  # symbol -> SharedSignals
        self.metadata = SharedMetadata()
        self.running = True
    
    def get_candles_handler(self, symbol: str) -> SharedCandles:
        """Get or create candles handler for symbol"""
        if symbol not in self.candles:
            self.candles[symbol] = SharedCandles(symbol)
        return self.candles[symbol]
    
    def get_indicators_handler(self, symbol: str) -> SharedIndicators:
        """Get or create indicators handler for symbol"""
        if symbol not in self.indicators:
            self.indicators[symbol] = SharedIndicators(symbol)
        return self.indicators[symbol]
    
    def get_signals_handler(self, symbol: str) -> SharedSignals:
        """Get or create signals handler for symbol"""
        if symbol not in self.signals:
            self.signals[symbol] = SharedSignals(symbol)
        return self.signals[symbol]
    
    def cleanup_symbol(self, symbol: str):
        """Cleanup shared memory for a specific symbol"""
        try:
            if symbol in self.candles:
                self.candles[symbol].cleanup()
                del self.candles[symbol]
            
            if symbol in self.indicators:
                self.indicators[symbol].cleanup()
                del self.indicators[symbol]
                
            if symbol in self.signals:
                self.signals[symbol].cleanup()
                del self.signals[symbol]
                
            self.logger.info(f"Cleaned up shared memory for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up {symbol}: {e}")
    
    def cleanup_all(self):
        """Cleanup all shared memory structures"""
        try:
            self.logger.info("Starting shared memory cleanup...")
            
            # Cleanup symbol-specific handlers
            for symbol in list(self.candles.keys()):
                self.cleanup_symbol(symbol)
            
            # Cleanup metadata
            self.metadata.cleanup()
            
            self.running = False
            self.logger.info("Shared memory cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage statistics"""
        try:
            stats = {
                'candles_count': len(self.candles),
                'indicators_count': len(self.indicators),
                'signals_count': len(self.signals),
                'symbols': list(self.candles.keys()),
                'total_handlers': len(self.candles) + len(self.indicators) + len(self.signals) + 1
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return {}