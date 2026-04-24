-- ============================================================
-- Helmet Detection Project - Database & Schema Setup
-- ============================================================

-- Create the main project database
CREATE DATABASE IF NOT EXISTS HELMET_DETECTION_DB;

-- Use the database
USE DATABASE HELMET_DETECTION_DB;

-- Create a dedicated warehouse for this project
CREATE WAREHOUSE IF NOT EXISTS HELMET_WH
    WITH WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- Schema for raw ingested data and metadata
CREATE SCHEMA IF NOT EXISTS RAW_DATA;

-- Schema for processed/transformed data (EDA outputs)
CREATE SCHEMA IF NOT EXISTS PROCESSED;

-- Schema for model artifacts, training runs, and registry
CREATE SCHEMA IF NOT EXISTS MODELS;

-- Schema for inference results and detection logs
CREATE SCHEMA IF NOT EXISTS RESULTS;
