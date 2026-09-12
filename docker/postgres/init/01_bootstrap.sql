\set ON_ERROR_STOP on
\set ECHO none
\set VERBOSITY terse
\getenv app_pw IP_BOOTSTRAP_APP_PASSWORD
\getenv etl_pw IP_BOOTSTRAP_ETL_PASSWORD
\getenv mcp_pw IP_BOOTSTRAP_MCP_PASSWORD
SET statement_timeout = '30s';
SET lock_timeout = '5s';

-- Never install dblink or postgres_fdw: they reopen a cross-database path and
-- invalidate the MCP credential boundary. vector is reserved for Phase 6.7.
-- The official entrypoint runs this file on an empty volume. bootstrap_db.sh
-- reruns the identical file against an existing volume without dropping data.
-- Serialize cluster-level creation across concurrent bootstrap invocations.
SELECT pg_advisory_lock(505, 5);
SELECT format('CREATE ROLE %I NOLOGIN', name)
FROM (VALUES ('app_owner'), ('biz_owner'), ('app_rw'), ('etl_rw'), ('mcp_ro')) AS roles(name)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = name) \gexec
ALTER ROLE app_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE biz_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE app_rw LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD :'app_pw';
ALTER ROLE etl_rw LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD :'etl_pw';
ALTER ROLE mcp_ro LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD :'mcp_pw';
-- Runtime roles must never inherit owner, predefined, or other runtime powers.
SELECT format('REVOKE %I FROM %I', parent.rolname, child.rolname)
FROM pg_auth_members membership
JOIN pg_roles parent ON parent.oid = membership.roleid
JOIN pg_roles child ON child.oid = membership.member
WHERE child.rolname IN ('app_rw', 'etl_rw', 'mcp_ro') \gexec

SELECT 'CREATE DATABASE insightpilot_app OWNER app_owner'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'insightpilot_app') \gexec
SELECT 'CREATE DATABASE insightpilot_business OWNER biz_owner'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'insightpilot_business') \gexec
ALTER DATABASE insightpilot_app OWNER TO app_owner;
ALTER DATABASE insightpilot_business OWNER TO biz_owner;
REVOKE ALL ON DATABASE insightpilot_app FROM PUBLIC, app_rw, etl_rw, mcp_ro;
REVOKE ALL ON DATABASE insightpilot_business FROM PUBLIC, app_rw, etl_rw, mcp_ro;
GRANT CONNECT ON DATABASE insightpilot_app TO app_rw;
GRANT CONNECT ON DATABASE insightpilot_business TO etl_rw, mcp_ro;
ALTER ROLE mcp_ro SET default_transaction_read_only = on;
ALTER ROLE mcp_ro SET statement_timeout = '10s';
ALTER ROLE mcp_ro SET idle_in_transaction_session_timeout = '15s';
ALTER ROLE mcp_ro SET lock_timeout = '2s';
-- \connect closes the administrative session and releases the advisory lock.
-- Database-local grants below are independently transactional and repeatable.
\connect insightpilot_app
SET statement_timeout = '30s';
SET lock_timeout = '5s';
BEGIN;
SELECT pg_advisory_xact_lock(505, 5);
ALTER SCHEMA public OWNER TO app_owner;
REVOKE ALL ON SCHEMA public FROM PUBLIC, app_rw, etl_rw, mcp_ro;
GRANT USAGE ON SCHEMA public TO app_rw;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC, app_rw, etl_rw, mcp_ro;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_rw;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC, app_rw, etl_rw, mcp_ro;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_rw;
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner
    REVOKE ALL ON TABLES FROM PUBLIC, app_rw, etl_rw, mcp_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner IN SCHEMA public
    REVOKE ALL ON TABLES FROM PUBLIC, app_rw, etl_rw, mcp_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner
    REVOKE ALL ON SEQUENCES FROM PUBLIC, app_rw, etl_rw, mcp_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner IN SCHEMA public
    REVOKE ALL ON SEQUENCES FROM PUBLIC, app_rw, etl_rw, mcp_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_rw;
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO app_rw;
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA public FROM PUBLIC;
COMMIT;

\connect insightpilot_business
SET statement_timeout = '30s';
SET lock_timeout = '5s';
BEGIN;
SELECT pg_advisory_xact_lock(505, 5);
CREATE SCHEMA IF NOT EXISTS biz AUTHORIZATION biz_owner;
ALTER SCHEMA biz OWNER TO biz_owner;
REVOKE ALL ON SCHEMA public FROM PUBLIC, app_rw, etl_rw, mcp_ro;
REVOKE ALL ON SCHEMA biz FROM PUBLIC, app_rw, etl_rw, mcp_ro;
GRANT USAGE ON SCHEMA biz TO etl_rw, mcp_ro;
REVOKE ALL ON ALL TABLES IN SCHEMA biz FROM PUBLIC, app_rw, etl_rw, mcp_ro;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA biz TO etl_rw;
GRANT SELECT ON ALL TABLES IN SCHEMA biz TO mcp_ro;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA biz FROM PUBLIC, app_rw, etl_rw, mcp_ro;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA biz TO etl_rw;
ALTER DEFAULT PRIVILEGES FOR ROLE biz_owner
    REVOKE ALL ON TABLES FROM PUBLIC, app_rw, etl_rw, mcp_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE biz_owner IN SCHEMA biz
    REVOKE ALL ON TABLES FROM PUBLIC, app_rw, etl_rw, mcp_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE biz_owner
    REVOKE ALL ON SEQUENCES FROM PUBLIC, app_rw, etl_rw, mcp_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE biz_owner IN SCHEMA biz
    REVOKE ALL ON SEQUENCES FROM PUBLIC, app_rw, etl_rw, mcp_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE biz_owner IN SCHEMA biz
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO etl_rw;
ALTER DEFAULT PRIVILEGES FOR ROLE biz_owner IN SCHEMA biz
    GRANT SELECT ON TABLES TO mcp_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE biz_owner IN SCHEMA biz
    GRANT USAGE, SELECT ON SEQUENCES TO etl_rw;
ALTER DEFAULT PRIVILEGES FOR ROLE biz_owner REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA biz FROM PUBLIC;
COMMIT;
