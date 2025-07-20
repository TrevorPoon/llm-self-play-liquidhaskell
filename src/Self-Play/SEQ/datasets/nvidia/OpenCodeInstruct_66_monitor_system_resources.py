from typing import *

def monitor_system_resources(interval=5, logfile='system_monitor.log'):
    """
    Continuously monitors the CPU and memory usage of the system and logs the 
    data to a file with timestamps every 'interval' seconds.

    :param interval: Time interval (in seconds) between log entries.
    :param logfile: Name of the log file.
    """
    with open(logfile, 'a') as file:
        while True:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            log_entry = f"{timestamp} - CPU Usage: {cpu_usage}% | Memory Usage: {memory_usage}%\n"
            file.write(log_entry)
            print(log_entry.strip())
            time.sleep(interval)

### Unit tests below ###
def check(candidate):
    assert candidate(interval=0.1, logfile='test_log_1.log') is None
    assert candidate(interval=0.1, logfile='test_log_2.log') is None
    assert candidate(interval=0.1, logfile='test_log_3.log') is None
    assert candidate(interval=0.1, logfile='test_log_4.log') is None
    assert candidate(interval=0.1, logfile='test_log_5.log') is None
    assert candidate(interval=0.1, logfile='test_log_6.log') is None
    assert candidate(interval=0.1, logfile='test_log_7.log') is None
    assert candidate(interval=0.1, logfile='test_log_8.log') is None
    assert candidate(interval=0.1, logfile='test_log_9.log') is None
    assert candidate(interval=0.1, logfile='test_log_10.log') is None

def test_check():
    check(monitor_system_resources)
