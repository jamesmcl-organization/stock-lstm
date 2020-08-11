from crontab import CronTab
import datetime

my_cron = CronTab (user=True)

job = my_cron.new (
    command='python /Users/jamesm/Desktop/Data_Science/stock_lstm/stock_lstm_pycharm/prepare_stock_data.py',
    comment='Daily Stock Information')

#Schedule Job for Monday-Friday at 4pm
job.setall('00 16 * * 1-5')

my_cron.write ()

for job in my_cron:
    sch = job.schedule (date_from=datetime.datetime.now ())
    print (sch.get_next ())
