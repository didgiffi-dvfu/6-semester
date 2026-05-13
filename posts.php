<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);
require_once('db.php');

$postId = $_GET['id']; 

$link = mysqli_connect($servername, $username, $password, $dbName);

if (!$link) {
    die('Ошибка подключения: ' . mysqli_connect_error());
}


$sql = "SELECT * FROM posts WHERE id=$postId";

$res = mysqli_query($link, $sql);

if (!$res) {
    die("Ошибка в SQL-запросе: " . mysqli_error($link));
}

// $rows = mysqli_fetch_array($res);
$title = 'Пост не найден';
$main_text = 'Запись не найдена или была удалена.';
$imagePath = '';

if ($res && mysqli_num_rows($res) > 0) {
    $rows = mysqli_fetch_array($res);
    $title = $rows['title'];
    $main_text = $rows['main_text'];
    $imagePath = $rows['image_path'];

    
}

mysqli_close($link);
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Избосарова Ж.О. - пост</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>

<body class="auth-page">
    <div class="container">
        <div class="pony-card mx-auto" style="max-width: 760px;">

            <div class="mb-4 text-center">
                <span class="badge rounded-pill bg-light text-secondary border px-3 py-2">
                    Пост #<?php echo $postId; ?>
                </span>
            </div>

            <div class="dashed-box mb-4 text-center">
                <h3 class="m-0">
                    <?php echo "<h1>$title</h1>"; ?>
                </h3>
            </div>

            <!-- <?php if ($imagePath !== ''): ?> -->
                <div class="post-img-container mb-4">
                    <img src="<?php echo $imagePath; ?>" alt="Изображение">
                    
                </div>
            <?php endif; ?>

            <p style="line-height: 1.8; font-size: 1.05rem;">
                <?php echo "<p>$main_text</p>"; ?>
                <!-- <?php echo "<p>$imagePath</p>"; ?> -->
            </p>

            <div class="d-flex justify-content-between align-items-center mt-4">
                <a href="/index.php" class="btn btn-outline-secondary rounded-pill px-4">← Назад</a>
                <a href="/profile.php" class="btn btn-pink px-4">В профиль</a>
            </div>

        </div>
    </div>
</body>
</html>