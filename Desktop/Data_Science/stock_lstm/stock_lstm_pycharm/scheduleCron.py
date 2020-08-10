from crontab import CronTab
my_cron = CronTab(user=True)


job = my_cron.new(command='python /Users/jamesm/Desktop/Data_Science/stock_lstm/stock_lstm_pycharm/prepare_stock_data.py')

job.minute.every(1)

job.minute.every(1)

my_cron.write()