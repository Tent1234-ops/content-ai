#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database Schema Migration: Fix notifications table to match SQLAlchemy model
Add missing columns: body, link
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
            # Get current columns
            cursor.execute("DESCRIBE notifications")
            columns = cursor.fetchall()
            column_names = [col['Field'] for col in columns]
            
            print('[INFO] Current notifications columns:')
            for col in column_names:
                print('  - ' + col)
            
            # Add missing columns
            columns_to_add = [
                ('body', "LONGTEXT NULL AFTER title"),
                ('link', "VARCHAR(255) NULL AFTER payload"),
            ]
            
            for col_name, col_def in columns_to_add:
                if col_name not in column_names:
                    sql = f"ALTER TABLE notifications ADD COLUMN {col_name} {col_def}"
                    try:
                        cursor.execute(sql)
                        connection.commit()
                        print(f'[SUCCESS] Added column: {col_name}')
                    except Exception as e:
                        print(f'[ERROR] Failed to add {col_name}: {e}')
                else:
                    print(f'[OK] Column already exists: {col_name}')
            
            # Verify final schema
            print('[INFO] Final notifications table structure:')
            cursor.execute("DESCRIBE notifications")
            columns = cursor.fetchall()
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
