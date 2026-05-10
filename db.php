<?php
$servername = '127.0.0.1';
$username = 'root';
$password = 'kali';
$dbName = 'mina_db';

$link = mysqli_connect($servername, $username, $password);

if (!$link) {
    die('Ошибка подключения: ' . mysqli_connect_error());
}

$sql = "CREATE DATABASE IF NOT EXISTS `$dbName`";

if (!mysqli_query($link, $sql)) {
    echo 'Не удалось создать БД';
}

mysqli_close($link);

$link = mysqli_connect($servername, $username, $password, $dbName);

if (!$link) {
    die('Ошибка подключения: ' . mysqli_connect_error());
}

mysqli_set_charset($link, 'utf8mb4');

$sql = "CREATE TABLE IF NOT EXISTS users(
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(15) NOT NULL UNIQUE,
    email VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(20) NOT NULL
)";

if (!mysqli_query($link, $sql)) {
    echo 'Не удалось создать таблицу Users';
}

$sql = "CREATE TABLE IF NOT EXISTS posts(
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(15) NOT NULL,
    title VARCHAR(100) NOT NULL,
    main_text TEXT NOT NULL,
    image_path VARCHAR(255) DEFAULT NULL
)";

if (!mysqli_query($link, $sql)) {
    echo 'Не удалось создать таблицу Posts';
}

mysqli_close($link);
?>
