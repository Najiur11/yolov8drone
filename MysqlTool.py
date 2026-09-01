import pandas as pd
import pymysql, json, datetime, configparser
from decimal import Decimal

class DroneDB:
    def __init__(self):
        cfg = configparser.ConfigParser()
        cfg.read('config.ini', encoding='utf-8')
        self.conn = pymysql.connect(
            host=cfg.get('DATABASE', 'host'),
            port=cfg.getint('DATABASE', 'port'),
            user=cfg.get('DATABASE', 'user'),
            password=cfg.get('DATABASE', 'password'),
            database=cfg.get('DATABASE', 'database'),
            charset='utf8mb4'
        )

    # 1) 保存轨迹到 drone_track
    def save_drone_data(self, data):
        """
        data: list[dict] 每项需包含
        drone_id, lng, lat, altitude, speed, status
        """
        sql = """
        INSERT INTO drone_track(drone_id, time, lng, lat, altitude, speed, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for d in data:
                    cur.execute(sql, (
                        d.get('drone_id', 1),
                        d.get('time', datetime.datetime.now()),
                        Decimal(str(d.get('lng', 0.0))),
                        Decimal(str(d.get('lat', 0.0))),
                        Decimal(str(d.get('altitude', 0.0))),
                        Decimal(str(d.get('speed', 0.0))),
                        d.get('status', '正常')
                    ))
            self.conn.commit()
            return True
        except Exception as e:
            print('[DroneDB]', e)
            return False
        finally:
            self.conn.close()

    # 2) 按无人机 ID 查询最新轨迹（调试用）
    def get_latest_track(self, drone_id=1, limit=100):
        sql = "SELECT * FROM drone_track WHERE drone_id=%s ORDER BY time DESC LIMIT %s"
        return pd.read_sql(sql, self.conn, params=(drone_id, limit))