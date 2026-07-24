#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database Schema Migration: Add missing 'body' column to notifications table
"""

import pymysql
import sys

def migrate_database():
    try:
        # Connect to database
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='1234',
            database='content_ai',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection.cursor() as cursor:
            # Check if column exists
            cursor.execute("DESCRIBE notifications")
            columns = cursor.fetchall()
            column_names = [col['Field'] for col in columns]
            
            if 'body' in column_names:
                print('[OK] Column body already exists in notifications table')
                return True
            
            # Add missing 'body' column
            sql = "ALTER TABLE notifications ADD COLUMN body LONGTEXT NULL AFTER title"
            cursor.execute(sql)
            connection.commit()
            print('[SUCCESS] Column body added to notifications table')
            
            # Verify
            cursor.execute("DESCRIBE notifications")
            columns = cursor.fetchall()
            print('[INFO] Notifications table structure:')
            for col in columns:
                print('  - ' + col['Field'] + ': ' + col['Type'])
            
            return True
            
    except pymysql.Error as e:
        print('[ERROR] MySQL Error: ' + str(e))
        return False
    except Exception as e:
        print('[ERROR] Unexpected error: ' + str(e))
        return False
    finally:
        if connection:
            connection.close()

if __name__ == '__main__':
    success = migrate_database()
    sys.exit(0 if success else 1)
